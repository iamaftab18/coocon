# Cocoon defect detection & automatic sorting

Raspberry Pi 4B · YOLOv8 · USB camera · relay-driven conveyor · L298N pusher motors

A tray of **2 × 4 cocoons** rides on a conveyor. A push button starts the cycle; the camera looks at one row
at a time, YOLOv8 labels each cocoon **Good** or **Bad**, and the two pusher motors push the bad ones out.

| File | What it is |
|---|---|
| `app.py` | The whole application: live preview, detection, sorting cycle, GPIO |
| `cocoon_training.ipynb` | Google Colab notebook that trains the YOLOv8 model from your Roboflow export |
| `requirements.txt` | Python packages |
| `dataset/` | Your raw photos (91 images, 640 × 480) |

**Roadmap:** &nbsp; A · Label in Roboflow &nbsp;→&nbsp; B · Export &nbsp;→&nbsp; C · Train in Colab &nbsp;→&nbsp; D · Run on the Raspberry Pi

```
 push button
     │
     ▼
 conveyor ON 2 s ─► OFF ─► settle ─► scan the row in the yellow STOP ZONE (3 frames vote)
                                              │
              ┌───────────────────────────────┴──────────────────────────────┐
         both Good                                                    at least one Bad
              │                                                               │
   conveyor ON until the next                                   conveyor ON 0.5 s (nudge)
   row reaches the stop zone                                    right one Bad → motor 1   (forward 4 s, back 4 s)
              ▲                                                 left one Bad  → motor 2   (forward 4 s, back 4 s)
              │                                                 both Bad      → both motors together
              └───────────────────────────────────────────────────────────────┘
                                 after 4 rows: "Tray complete"
```

---

# Part A · Label your images in Roboflow

## A0. Decide what "Bad" means — before you draw a single box

The model learns *exactly* what you label, so the rule must be written down and used the same way on every image.
Edit this table to match your quality rules (the "seen in your photos" column is what I noticed in your dataset —
please confirm, I don't know your grading rules):

| Cocoon | Seen in your photos | Label (edit me) |
|---|---|---|
| Clean, intact, white / cream | ✔ many | **Good** |
| Golden / brownish, uniform | ✔ many | *your decision* — natural colour, or a stain defect? |
| Small dark hole (pierced) | ✔ e.g. a golden cocoon with a black dot near its top | **Bad** |
| Torn / open end | ✔ a white cocoon whose tip is open, in many photos | *your decision* |
| Loose fluff / floss on an otherwise fine cocoon | ✔ some | *your decision* |
| Dirty / stained patch, misshapen, double cocoon | – | **Bad** if it is a defect for you |

Exactly **two classes**, named exactly **`Good`** and **`Bad`** (`app.py` refuses to start with other names).

## A1. Health-check the photos you have

I looked at a sample of your 91 photos. Good news: same resolution, same top-down view, tray clearly visible.
Things that will limit how *trustworthy* the numbers are:

1. **Near-duplicates.** Many files have the same timestamp second (`…_23_30_20_Pro (2).jpg`, `(3)`, `(4)`) and the same
   few cocoons appear again and again in slightly different positions. Roboflow splits randomly, so almost identical frames end up in
   both *train* and *valid* — the validation score will look better than real life.
2. **Position and colour are tied together.** In every photo I opened, the **left column is white and the right column golden**.
   If your rule is "golden = Bad" that is fine, but then the model might partly learn *"right side = Bad"*. When you take more photos,
   **shuffle** which cocoon sits left or right, and rotate/flip them.
3. **Pen marks on the tray** (e.g. a "G" next to a hole). If they mean Good/Bad, cover them or use a clean tray, otherwise the model can
   learn the letter instead of the cocoon.
4. **How many?** 91 photos is enough for a first working model. For a machine you can rely on, aim for **200 – 300 photos with 20+ different
   cocoons per class**, different positions, and the lighting you will really use. More *different cocoons* beats more epochs.

## A2. Create the project and upload

1. Sign up / log in at **roboflow.com** (free plan is enough; on the free plan projects are public — fine for cocoon photos).
2. **Create New Project** → name `cocoon-defects` → project type **Object Detection** → annotation group `cocoon`.
3. **Upload Data** → drag in all images from the `dataset` folder → **Save and Continue** (wait until all 91 are uploaded).

## A3. Draw the boxes

Open **Annotate** and start with the first image. In the editor pick the **bounding-box tool**, drag a box around a cocoon,
type the class name in the pop-up (`Good` / `Bad` — create both the first time, pick from the list afterwards) and press **Enter**.
Move to the next image (arrow / next button); Roboflow saves as you go. About 4 boxes per image → 20–30 minutes for all photos.

Rules that keep the labels clean:

- **One box per cocoon**, as tight as possible around the *visible cocoon* (not the tray hole, not the shadow).
- **Label every cocoon that is more than half visible** — including the ones cut off at the top/bottom edge of the frame (rows entering / leaving).
  Skip cocoons that are less than half visible. Never leave a clearly visible cocoon unlabeled — the model would learn that it is background.
- **No box on empty holes.**
- **When unsure, zoom in and decide by your rule table** (A0). If you truly can't decide, delete that photo from the project rather than guess.
- Same rule on the first and on the last image. Consistency matters more than perfection.

Quick self-check when done: open **Dataset ▸ Health Check** (class balance). Both classes should have plenty of boxes;
if one class has under ~30, photograph more of it.

## A4. Add to the dataset and split

At the top of the Annotate page click **Add … Images to Dataset**, choose **Split Images Between Train/Valid/Test**, and set roughly
**Train 80 % / Valid 20 % / Test 0 %**. (With this little data a 9-image test split tells you nothing; instead test on *new* cocoons — see Part C.
If Roboflow won't accept 0 % test, keep the default 70 / 20 / 10 — the notebook handles both.)

---

# Part B · Export from Roboflow

## B1. Create a dataset version

1. Sidebar ▸ **Versions** (older UI: *Generate*) ▸ **Create New Version**.
2. **Preprocessing:** keep **Auto-Orient**, and **delete the "Resize" step** (Roboflow adds *Stretch to 640×640* by default).
   *Why:* stretching squashes your 4:3 photos into squares. The Pi feeds normal 4:3 frames, so the model would see cocoons with different
   proportions at run time than during training. Ultralytics resizes correctly (without distortion) by itself.
3. **Augmentation:** add **none**. YOLOv8 already applies strong random augmentation (mosaic, flips, rotation, brightness/colour) on every epoch.
4. **Create.**

## B2. Export

On the version page click **Export Dataset** (or *Download Dataset*), choose the format **YOLOv8**, then either:

- **Download zip to computer** → you will upload the `.zip` in Colab (*Option "upload zip"*), or
- **Show download code** → note the three names in the snippet (`workspace`, `project`, `version`) and your **Private API key**
  (Roboflow ▸ Settings ▸ API Keys) → used in Colab (*Option "roboflow api"*).

---

# Part C · Train in Google Colab

1. Go to **colab.research.google.com** ▸ **File ▸ Upload notebook** ▸ choose `cocoon_training.ipynb`.
2. **Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save.**
3. Run the cells from top to bottom (**Runtime ▸ Run all** works too):
   - **Cell "Get the dataset":** set `SOURCE` — *upload zip* (a file picker opens; choose the Roboflow zip) or *roboflow api*
     (fill `WORKSPACE`, `PROJECT`, `VERSION`; the API key is asked in a hidden prompt).
   - **Cell "Check the dataset":** confirms both classes are called `Good` / `Bad`, prints images and boxes per split.
   - **Cell "Train":** YOLOv8n, 416 px, batch 8, up to 60 epochs with early stopping — a few minutes on the T4 GPU.
   - **Cells "How good is it?" and "Predictions":** metrics, confusion matrix and pictures.
   - **Cell "Export + download":** downloads **`cocoon_model.zip`** with **`best_ncnn_model/`** (the model the Raspberry Pi runs) and `best.pt`
     (the PyTorch original, for a PC). If the NCNN conversion says *skipped*, run that cell again — it needs internet access in Colab.
4. **Reading the result**

   | Number | Meaning | Good value |
   |---|---|---|
   | mAP50 | overall detection quality | above 0.90 |
   | Precision (Bad) | of cocoons called Bad, how many really are — low = good cocoons get pushed out | above 0.90 |
   | Recall (Bad) | of the really Bad cocoons, how many are found — low = defects slip through | above 0.90 |

   ⚠ Values very close to 1.00 on this dataset are **not proof** (near-duplicate photos, see A1). Use the optional
   *"try a photo the model has never seen"* cell with photos of **different cocoons**, and later judge the real machine.
5. **If it is not accurate enough** — in this order:
   1. Look at the prediction pictures: wrong or inconsistent *labels* are the most common cause. Fix them in Roboflow, create a new version, retrain.
   2. Add more photos of **different** cocoons, especially of the class that performs worse.
   3. Increase `IMG_SIZE` (480 or 640) — slower on the Pi, and you must set the same value in `app.py`.
   4. Use `yolov8s.pt` — slower on the Pi.
   5. More epochs (rarely the answer; early stopping already picks the best epoch).

---

# Part D · Run it on the Raspberry Pi 4B

## D1. What you need

- Raspberry Pi 4B (2 GB or more) with **Raspberry Pi OS 64-bit** *with desktop* (Bookworm or newer)
- **USB webcam** (the kind you used for the dataset), monitor + keyboard (or VNC) for the preview window.
  A Raspberry Pi *Camera Module* (ribbon cable) needs a different library (`picamera2`) and is **not** supported by this `app.py`
- 5 V **active-low** relay module for the conveyor · **L298N** motor driver · 2 pusher motors · push button · separate motor power supply

## D2. Install

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-gpiozero python3-lgpio python3-yaml

# virtual environment that can also see the GPIO libraries installed by apt
python3 -m venv --system-site-packages ~/cocoon-env
source ~/cocoon-env/bin/activate

cd ~/cocoon            # the folder with app.py, requirements.txt and the best_ncnn_model folder
pip install --upgrade pip
pip install -r requirements.txt        # about a minute
```

Put the **`best_ncnn_model`** folder (from `cocoon_model.zip`) next to `app.py` — that is the model the app loads by default.
The Pi runs it with the small `ncnn` package: **no PyTorch, no Ultralytics** (their prebuilt wheels crash with *Illegal instruction* on a
Raspberry Pi 4, see Troubleshooting). `best.pt` is only for a PC. The on-screen `det … fps` shows the speed you really get.

## D3. First run — detection only

Do this **before wiring any motors**, from a terminal on the Pi's own desktop (not plain SSH — the window needs the display):

```bash
source ~/cocoon-env/bin/activate
python app.py
```

You should see the camera picture, the yellow **STOP ZONE**, and green / red boxes labelled `GOOD` / `BAD` with confidence. Hold a tray in
view. Check that both cocoons of a row are boxed and labelled correctly. Keys: **S** start, **X** emergency stop, **Q** quit.

`app.py` also runs on a Windows/Linux PC with a webcam (`--source 0`, or a video file): there the GPIO is simulated and a yellow note says so.

## D4. Wiring (BCM numbers = GPIO numbers)

| Function | Pi GPIO | Pi pin | Goes to |
|---|---|---|---|
| Push button | GPIO 17 | 11 | button, other leg to GND (pin 9) — internal pull-up is used |
| Conveyor relay `IN` | GPIO 4 | 7 | relay module `IN` (active-low: LOW = conveyor ON) |
| Relay `VCC` / `GND` | 5 V / GND | 2 / 6 | relay module supply |
| L298N `IN1` / `IN2` | GPIO 16 / 20 | 36 / 38 | **motor 1** = pusher for the **RIGHT** cocoon (`OUT1`/`OUT2`) |
| L298N `IN3` / `IN4` | GPIO 21 / 26 | 40 / 37 | **motor 2** = pusher for the **LEFT** cocoon (`OUT3`/`OUT4`) |
| L298N `GND` | GND | 34 | **must be connected to the Pi GND** *and* the motor supply GND |

- Leave the **ENA / ENB jumpers on** and don't connect anything to them (then motors always run at full speed; set `MOTOR_SPEED` in `app.py` to reduce it).
  If you remove the jumpers and wire ENA/ENB to GPIO 12/13, set `enable=12` / `enable=13` in `MOTORS`.
- Power the L298N motor terminal from its **own supply** (pusher motor voltage). Do **not** connect the L298N `+5V` pin to the Pi.
- The relay's contacts (`COM`/`NO`) switch the conveyor motor's own supply. If that is mains voltage, have a qualified person do the wiring and enclose it.
- GPIO 4 is used for the relay because it idles HIGH while the Pi boots, so an active-low relay does not click on at power-up. Some relay modules
  can still float when the program exits — a 10 k pull-up from `IN` to 3.3 V fixes that.
- Direction wrong? Swap the two motor wires (or the two pin numbers in `MOTORS`). Relay turns **on** when idle? Set `RELAY_ACTIVE_LOW = False`.
- **Add a hardware emergency stop** that cuts motor power. The `X` key is only a software stop.

**Bench-test the outputs** (motors unloaded) with this snippet — the relay should click once, then each motor runs 1 s forward and 1 s back:

```bash
python - <<'EOF'
from gpiozero import OutputDevice, Motor
from time import sleep
relay = OutputDevice(4, active_high=False, initial_value=False)
relay.on(); sleep(1); relay.off()
for pins in ((16, 20), (21, 26)):
    m = Motor(*pins); m.forward(); sleep(1); m.backward(); sleep(1); m.stop()
EOF
```

## D5. Calibrate (do this once, in this order)

| Step | What to do |
|---|---|
| **1 · Stop zone** | Put a tray on the belt, run `python app.py`, push a row by hand to where the pushers need it. Adjust `ROW_CENTER_Y` so the **centres of both cocoons** sit inside the yellow zone, ideally in its middle (a slightly tilted tray is fine). The default is the middle of the picture — in some of your photos that is the *gap between* two rows, so **this step matters**. The zone (`2 × ROW_TOLERANCE` of the image height) must be clearly **smaller than the distance between two rows** (about 0.45 of the height in your photos). |
| **2 · Left / right** | Put a Bad cocoon on the right: it must be labelled BAD on the right side of the picture and fire **motor 1**. Mirrored? `SWAP_LEFT_RIGHT = True`. |
| **3 · `FIRST_MOVE_S`** | Start a tray and look after the first move: the first row must **not have passed** the stop zone yet (it may be short of it — the app then creeps forward to find it). Shorten it if row 1 is already beyond the zone. |
| **4 · `NUDGE_S`** | Time the conveyor needs to bring a cocoon from the stop zone to the pushers (default 0.5 s). |
| **5 · Belt speed** | Use `det … fps` on screen. In my simulations the sorting was reliable whenever the belt needed **about 2 s or more to move one row spacing**; with fast detection (10 fps) even ~1.2 s worked. Faster than that, the app stops with a *"row not found"* error instead of mis-sorting — slow the belt, or train/export the model at a smaller `IMG_SIZE`. |
| **6 · `CONF_THRESHOLD`** | 0.5 by default. Raise it if false detections appear, lower it if real cocoons are missed. |

Run a **dry cycle first**: no cocoons in the pusher path, watch the state line and the `Conveyor / M1 / M2` line on screen, then a real tray.

## D6. Every-day use

1. `source ~/cocoon-env/bin/activate && python app.py`
2. Place a tray, press the push button (or **S**).
3. Watch the status. When "Tray complete" shows, remove the tray and place the next one.

---

# Reference

## How a row is handled

- **Stop zone** = the yellow band. Left / right cocoon = position in the image.
- After the button (or after a row is finished) the conveyor runs until a **new row is complete in the zone**: both cocoons inside it, at least one
  of them not scanned before. Cocoons are followed with small ids, so a row that was just handled is never mistaken for the next one, and a row is
  never skipped, whatever the belt speed. If only one cocoon shows up (tilted tray, a missed detection) the belt keeps running for `ROW_WAIT_S`
  and then stops anyway.
- The conveyor stops, waits `SETTLE_S` for a sharp image, then **3 frames** that show both cocoons vote (on a tie, *Bad* wins).
- **Both Good →** next row. **Any Bad →** nudge (`NUDGE_S`), then motor 1 (right) and/or motor 2 (left) forward `PUSH_FORWARD_S`, back `PUSH_BACK_S`;
  then the pusher retracts, the app re-checks the row with the belt stopped, and moves on.
- After `ROWS_PER_TRAY` (4) rows: *Tray complete*.
- **Anything unexpected** (no row found, detector stalled for 4 s while the belt runs, camera lost, no row within 12 s) switches the conveyor and motors
  **off** and shows a red **ERROR** — press the button to start again. Restarting counts from row 1 again.
- **Emergency stop (`X`)** switches everything off *where it is*: a pusher may be left extended — retract it by hand before restarting.

## Settings you may change (top of `app.py`)

`MODEL_PATH` · `IMG_SIZE` (= training size) · `CONF_THRESHOLD` · `CAMERA_INDEX` · `ROW_CENTER_Y` · `ROW_TOLERANCE` · `MIN_COLUMN_GAP` · `SWAP_LEFT_RIGHT` ·
`FIRST_MOVE_S` · `SETTLE_S` · `SCAN_FRAMES` · `ROW_WAIT_S` · `NUDGE_S` · `PUSH_FORWARD_S` · `PUSH_BACK_S` · `ROWS_PER_TRAY` · `BUTTON_PIN` · `RELAY_PIN` ·
`RELAY_ACTIVE_LOW` · `MOTOR_SPEED` · `MOTORS`

## Troubleshooting

| Symptom | Fix |
|---|---|
| **`Illegal instruction`** (the program just dies) | PyTorch's prebuilt wheels use CPU instructions that the Pi 4's Cortex-A72 does not have ([PyTorch #132032](https://github.com/pytorch/pytorch/issues/132032), [#174344](https://github.com/pytorch/pytorch/issues/174344)). `app.py` now runs the NCNN model without PyTorch: put the `best_ncnn_model` folder next to it, `pip install ncnn pyyaml`, and run `python app.py`. Never start it with `--model best.pt` on a Pi 4. Still dying? `faulthandler` prints the Python line it died on — if it is `import cv2` / `import numpy`, use apt's versions: `pip uninstall -y opencv-python numpy` and `sudo apt install python3-opencv python3-numpy`. |
| `Missing package (ncnn)` | `pip install ncnn pyyaml` (inside the virtual environment). |
| `Model not found: 'best_ncnn_model'` | Copy the `best_ncnn_model` folder from `cocoon_model.zip` next to `app.py`, or `--model path`. |
| `The model has classes [...] but app.py expects 'good' and 'bad'` | Rename the classes in Roboflow to `Good` / `Bad`, new version, retrain. |
| `GPIO library not found` (on the Pi) | `sudo apt install python3-lgpio`, and create the virtual environment with `--system-site-packages` (D2). The app deliberately refuses to "simulate" on a real Pi. |
| `Cannot open camera` | `ls /dev/video*`; try `--source 1`; unplug/replug; close other programs using the camera. |
| Window does not open / `xcb` or `Authorization required` | Run from a terminal on the Pi's desktop, or `export DISPLAY=:0` first. Still failing: `sudo apt install -y libxcb-xinerama0`, or use the system OpenCV: `pip uninstall -y opencv-python` (with `python3-opencv` installed via apt). |
| Slow (`det` only 1-3 fps) | Close other programs; train and export at a smaller size (e.g. 320) in the notebook — the app follows the size stored in the model. |
| `ERROR: row N not found in the stop zone after stopping (best frame: K of 2 cocoons)` | **K = 0:** the row stopped outside the yellow zone (belt too fast for the detection speed, or zone in the wrong place) — slow the belt / move `ROW_CENTER_Y`. **K = 1:** one cocoon is not detected with enough confidence (watch its `conf` in the preview; try a lower `CONF_THRESHOLD`, better light, or more training photos of that kind of cocoon). |
| `ERROR: no cocoon reached the stop zone in 12 s` | No tray in the camera's view, or the stop zone is in the wrong place (D5, step 1). |
| `ERROR: row N never arrived …` | Tray has fewer rows than `ROWS_PER_TRAY`, a slot is empty (every slot needs a cocoon), or `FIRST_MOVE_S` was so long that row 1 was skipped. |
| `ERROR: detector gave no results` / `detector stopped delivering frames` | The model crashed or is far too slow — check the terminal output. |
| Relay on when idle / motor spins the wrong way | `RELAY_ACTIVE_LOW`, or swap the motor wires (see D4). |
| Motor does nothing | L298N motor supply, common GND, ENA/ENB jumpers on, `MOTORS` pin numbers. |
| `GPIO busy` / pin in use | Another program is using the pin — close it (`pkill -f app.py`). |

## Known limits

- **Every slot needs a cocoon.** A row is "complete" when two cocoons are seen; an empty hole makes the row unreadable (→ error stop).
- The conveyor is switched on/off (no speed control), so the row stops slightly *after* it is first seen; the yellow zone is sized for that (D5).
- The **first** row must still be before/inside the zone after `FIRST_MOVE_S`; if it has already passed, the app cannot know and treats the next row as row 1
  (it then errors at the end because a row is missing).
- Sorting quality is only as good as the labels and the model (Part A / C).

## Testing status (honest summary)

Tested on a PC: the sorting state machine in a simulator (fake belt, tilted trays, mock GPIO, simulated detection noise, pusher scenarios, error paths);
the whole app running in real time with real threads and a moving belt built from your photos (4 rows sorted, motors fired as decided); and the
notebook's code cells. The NCNN backend was checked against Ultralytics on **your trained model over all 91 photos**: identical detections to
Ultralytics' own NCNN loader (0.00 px, 0.0000 confidence difference), and it loads no PyTorch or Ultralytics at all. The belt runs used a throw-away
model or your model on a synthetic belt, so they prove the *plumbing*, not the Good/Bad accuracy — that depends on your labels (Part A).
**Not tested by me:** the real Raspberry Pi (the `ncnn` package and the rest of this stack were only analysed, not run on a Cortex-A72),
GPIO / relay / L298N / belt, the Pi's frame rate, and the Roboflow and Colab web pages (steps written from experience — their menus move around
a little over time; look for the equivalent button).
