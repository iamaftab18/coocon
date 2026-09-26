#!/usr/bin/env python3
"""
Cocoon defect detection + automatic sorting  (Raspberry Pi 4B)

Live camera preview with YOLOv8 detection, plus a push-button controlled sorting cycle:

  button -> conveyor runs FIRST_MOVE_S -> scan the row that sits in the yellow STOP ZONE
      both Good -> conveyor runs until the next row reaches the zone
      a Bad one -> conveyor nudge (NUDGE_S), then motor 1 (RIGHT cocoon) and/or motor 2 (LEFT cocoon)
                   run forward PUSH_FORWARD_S and back PUSH_BACK_S, then on to the next row
  After ROWS_PER_TRAY rows the tray is finished.

Left / right = position in the camera image (see SWAP_LEFT_RIGHT). If anything is unclear
(row not found, detector or camera failure) everything is switched off and an ERROR is shown
instead of guessing.

Keys in the preview window:  S = start   X = emergency stop   Q = quit
Run:  python app.py            (options: --model best.pt --source 0)
"""
import argparse
import os
import sys
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

if sys.platform.startswith("linux"):
    # pip's OpenCV can't open its window under Wayland unless Qt is told to use X11
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# =============================================================================
#  SETTINGS  -  everything you may need to tune is here
# =============================================================================

# ---- model ------------------------------------------------------------------
MODEL_PATH = "best.pt"        # or the NCNN folder "best_ncnn_model" (faster on the Pi)
IMG_SIZE = 416                # must equal the size used for training / export
CONF_THRESHOLD = 0.50         # ignore detections less confident than this
GOOD_NAME, BAD_NAME = "good", "bad"   # class names from Roboflow (case-insensitive)

# ---- camera -----------------------------------------------------------------
CAMERA_INDEX = 0              # USB webcam. 0 = first camera
FRAME_W, FRAME_H = 640, 480

# ---- where a row must be to count as "in position" -----------------------------
ROW_CENTER_Y = 0.50           # centre of the stop zone, as a fraction of image height
ROW_TOLERANCE = 0.12          # zone = centre +/- this fraction (0.12 -> 24 % of the height);
                              #   the zone must be clearly SMALLER than the spacing between two rows
MIN_COLUMN_GAP = 0.20         # left/right cocoon must be at least this far apart (fraction of width)
SWAP_LEFT_RIGHT = False       # True if the camera is mounted so left/right come out mirrored

# ---- timing, seconds --------------------------------------------------------
FIRST_MOVE_S = 2.0            # conveyor run after the button press, before the first scan
                              #   (the first row must NOT have passed the stop zone by then)
ADVANCE_TIMEOUT_S = 12.0      # next row must reach the stop zone within this time, else error
SETTLE_S = 0.6                # wait after the conveyor stops so the image is sharp
SCAN_FRAMES = 3               # frames showing a complete row that are voted on (good / bad)
ROW_WAIT_S = 0.5              # a new cocoon is in the zone but its partner is not (tilted tray, missed
                              #   detection): keep the belt running this long, then stop anyway
SCAN_TIMEOUT_S = 8.0          # give up scanning after this long
DETECTOR_STALL_S = 4.0        # conveyor running and no fresh detection for this long -> stop with an error
NUDGE_S = 0.5                # extra conveyor run to bring a bad cocoon to the pushers
PUSH_FORWARD_S = 4.0          # motor forward
PUSH_BACK_S = 4.0             # motor backward
ROWS_PER_TRAY = 4             # 2 x 4 tray = 4 rows

# ---- GPIO, BCM numbers ------------------------------------------------------
BUTTON_PIN = 17               # push button between this pin and GND
RELAY_PIN = 4                 # conveyor relay (GPIO4 idles HIGH at power-up -> an active-low relay stays off)
RELAY_ACTIVE_LOW = True       # True: pin LOW = conveyor ON.  Set False if your relay turns ON when idle.
MOTOR_SPEED = 1.0             # 0.0 - 1.0
# L298N.  Motor 1 pushes the RIGHT cocoon, motor 2 pushes the LEFT cocoon.
# Keep the ENA/ENB jumpers on the L298N and leave enable=None.
# (If you removed the jumpers and wired ENA/ENB to GPIO 12/13, set enable=12 / enable=13.)
MOTORS = {
    1: dict(forward=16, backward=20, enable=None),   # IN1, IN2  (right)
    2: dict(forward=21, backward=26, enable=None),   # IN3, IN4  (left)
}

GOOD_NAME, BAD_NAME = GOOD_NAME.lower(), BAD_NAME.lower()
SCAN_FRAMES = max(1, SCAN_FRAMES)
WINDOW = "Cocoon Sorter"
GREEN, RED, YELLOW, WHITE, GREY = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255), (170, 170, 170)


# =============================================================================
#  Data types + row logic (no hardware, easy to test)
# =============================================================================
@dataclass
class Det:
    label: str      # "good" or "bad" (lower-case)
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    tid: int = 0    # track id, see RowTracker

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


@dataclass
class Result:
    frame_id: int
    t: float                      # time the camera frame was grabbed (monotonic seconds)
    w: int
    h: int
    dets: list = field(default_factory=list)


def zone_dets(res):
    """Detections whose centre lies inside the stop zone, most confident first."""
    lo, hi = (ROW_CENTER_Y - ROW_TOLERANCE) * res.h, (ROW_CENTER_Y + ROW_TOLERANCE) * res.h
    return sorted((d for d in res.dets if lo <= d.cy <= hi), key=lambda d: -d.conf)


def find_row(res):
    """Return (left, right) detections if a complete row is inside the stop zone, else None."""
    in_zone = zone_dets(res)
    if not in_zone:
        return None
    first = in_zone[0]
    for other in in_zone[1:]:
        if abs(other.cx - first.cx) >= MIN_COLUMN_GAP * res.w:
            left, right = sorted((first, other), key=lambda d: d.cx)
            return (right, left) if SWAP_LEFT_RIGHT else (left, right)
    return None


class RowTracker:
    """Gives every cocoon an id that survives a few missed detections while the belt moves.
    That is how the sorter knows a cocoon is NEW and not one it has already handled
    (so rows are never scanned twice and never skipped, whatever the belt speed)."""
    TTL_S = 2.0        # while the belt moves: forget a cocoon not seen for this long
    GATE = 0.20        # max distance a cocoon may move between two detections (fraction of image height)

    def __init__(self):
        self.tracks = {}            # id -> (cx, cy, time last seen)
        self.next_id = 1

    def reset(self):
        self.tracks = {}

    def update(self, res, moving):
        if moving:     # with the belt stopped a hidden cocoon (e.g. behind a pusher) keeps its id
            self.tracks = {i: t for i, t in self.tracks.items() if res.t - t[2] <= self.TTL_S}
        free, gate2 = set(self.tracks), (self.GATE * res.h) ** 2
        for d in sorted(res.dets, key=lambda d: -d.conf):
            def dist2(i):
                return (self.tracks[i][0] - d.cx) ** 2 + (self.tracks[i][1] - d.cy) ** 2
            best = min(free, key=dist2, default=None)         # nearest cocoon we already know
            if best is not None and dist2(best) <= gate2:
                d.tid = best
                free.discard(best)
            else:                                             # a new cocoon
                d.tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[d.tid] = (d.cx, d.cy, res.t)


# =============================================================================
#  Hardware: conveyor relay, L298N motors, push button
# =============================================================================
def on_raspberry_pi():
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"raspberry pi" in f.read().lower()
    except OSError:
        return False


class Hardware:
    def __init__(self):
        from gpiozero.exc import BadPinFactory, PinFactoryFallback
        warnings.filterwarnings("ignore", category=PinFactoryFallback)   # noisy on a PC, irrelevant on a Pi
        self.start_event = threading.Event()
        self.simulated = False
        try:
            self._setup()
        except BadPinFactory:
            if on_raspberry_pi():         # never "simulate" on the real machine - the conveyor would not move
                sys.exit("GPIO library not found. Install it with:  sudo apt install python3-lgpio\n"
                         "and create the virtual environment with:  python3 -m venv --system-site-packages ~/cocoon-env")
            # Not a Raspberry Pi (e.g. testing on a PC): use fake pins so the app still runs.
            from gpiozero import Device
            from gpiozero.pins.mock import MockFactory, MockPWMPin
            Device.pin_factory = MockFactory(pin_class=MockPWMPin)
            self.simulated = True
            self._setup()

    def _setup(self):
        from gpiozero import Button, Motor, OutputDevice
        # active-low relay: .on() drives the pin LOW (relay ON), .off() drives it HIGH (relay OFF).
        # initial_value=False -> starts OFF.
        self.relay = OutputDevice(RELAY_PIN, active_high=not RELAY_ACTIVE_LOW, initial_value=False)
        self.motors = {i: Motor(pwm=True, **pins) for i, pins in MOTORS.items()}
        self.button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.05)
        self.button.when_pressed = self.start_event.set
        self.motor_dir = {i: "-" for i in MOTORS}

    # -- conveyor
    def conveyor(self, on):
        if on:
            self.relay.on()
        else:
            self.relay.off()

    @property
    def conveyor_on(self):
        return self.relay.is_active

    # -- pusher motors
    def motors_forward(self, ids):
        for i in ids:
            self.motors[i].forward(MOTOR_SPEED)
            self.motor_dir[i] = "FWD"

    def motors_backward(self, ids):
        for i in ids:
            self.motors[i].backward(MOTOR_SPEED)
            self.motor_dir[i] = "BACK"

    def motors_stop(self):
        for i, m in self.motors.items():
            m.stop()
            self.motor_dir[i] = "-"

    def all_off(self):
        self.conveyor(False)
        self.motors_stop()

    # -- button
    def start_pressed(self):
        if self.start_event.is_set():
            self.start_event.clear()
            return True
        return False

    def status(self):
        return "Conveyor:%s  M1:%s  M2:%s" % ("ON" if self.conveyor_on else "off",
                                               self.motor_dir[1], self.motor_dir[2])

    def close(self):
        self.all_off()
        for dev in (self.relay, self.button, *self.motors.values()):
            dev.close()


# =============================================================================
#  YOLO detector running in a background thread (keeps the preview smooth)
# =============================================================================
class Detector(threading.Thread):
    def __init__(self, model_path):
        super().__init__(daemon=True)
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        # the first prediction is slow (lazy initialisation): do it now, so "Ready" really means ready
        self.model.predict(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), imgsz=IMG_SIZE, verbose=False)
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._frame_id = 0
        self._quit = threading.Event()
        self.result = None            # latest Result
        self.error = None
        self.fps = 0.0

    def submit(self, frame, t):
        with self._lock:
            self._frame, self._t, self._frame_id = frame, t, self._frame_id + 1

    def stop(self):
        self._quit.set()

    def run(self):
        last_id = 0
        try:
            while not self._quit.is_set():
                with self._lock:
                    frame, t, fid = self._frame, self._t, self._frame_id
                if frame is None or fid == last_id:
                    time.sleep(0.005)
                    continue
                last_id = fid
                t0 = time.monotonic()
                h, w = frame.shape[:2]
                out = self.model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, agnostic_nms=True,
                                         max_det=10, verbose=False)[0]
                dets = [Det(self.names[int(c)], float(p), *map(float, xyxy))
                        for xyxy, p, c in zip(out.boxes.xyxy.tolist(), out.boxes.conf.tolist(),
                                              out.boxes.cls.tolist())]
                self.result = Result(fid, t, w, h, dets)
                dt = time.monotonic() - t0
                self.fps = 0.8 * self.fps + 0.2 / dt if self.fps else 1 / dt
        except Exception as exc:      # surface the error in the main loop so it can stop the conveyor
            self.error = exc


# =============================================================================
#  Sorting state machine (non-blocking, so the preview never freezes)
# =============================================================================
class Sorter:
    def __init__(self, hw, clock=time.monotonic):
        self.hw, self.clock = hw, clock
        self.state = "IDLE"
        self.msg = "Press the button to start"
        self.rows_done = 0
        self.n_good = self.n_bad = 0
        self.last_row = ""
        self.t0 = clock()
        self.timer = 0.0              # seconds the current state lasts (0 = open-ended)
        self.seeking_first = False    # True until the first row has been scanned
        self.sought = False           # the one allowed "creep forward to find the first row" was used
        self.tracker = RowTracker()
        self.tracked_frame = -1
        self.done_ids = set()         # track ids of the cocoons that were already scanned
        self.first_seen = None        # ADVANCE: frame time when the first new cocoon entered the zone
        self.scan_ids = set()         # ids seen in the stop zone while scanning the current row
        self.targets = []             # motors to fire for the current row
        self.votes = []               # (left, right) labels from frames with a complete row
        self.misses = 0               # scanned frames without a complete row
        self.seen = 0                 # frames looked at while re-checking a row after a push
        self.last_id = -1             # id of the last detector frame this state has looked at

    # -- helpers
    def _enter(self, state, msg, timer=0.0):
        self.state, self.msg, self.timer, self.t0 = state, msg, timer, self.clock()
        self.last_id = -1

    def _fresh(self, res):
        """True once per detector frame that was grabbed after the current state began."""
        if res and res.t >= self.t0 and res.frame_id != self.last_id:
            self.last_id = res.frame_id
            return True
        return False

    def remaining(self):
        return max(0.0, self.timer - (self.clock() - self.t0)) if self.timer else None

    def stop(self, msg="Stopped"):
        self.hw.all_off()
        self._enter("IDLE", msg)

    def _fail(self, msg):
        self.stop(msg)
        self.state = "ERROR"

    def _next_row(self):
        self.rows_done += 1
        if self.rows_done >= ROWS_PER_TRAY:
            self.hw.all_off()
            self._enter("IDLE", "Tray complete (%d rows). Load next tray and press the button." % self.rows_done)
        else:
            self._advance("Moving to row %d" % (self.rows_done + 1))

    def _advance(self, msg):
        """Run the conveyor until a new row (cocoons that were not scanned yet) is in the stop zone."""
        self.first_seen = None
        self.hw.conveyor(True)
        self._enter("ADVANCE", msg)

    def _stop_and_settle(self):
        self.hw.conveyor(False)
        self._enter("SETTLE", "Row %d in position, settling" % (self.rows_done + 1), SETTLE_S)

    # -- public
    def start(self):
        self.rows_done = 0
        self.n_good = self.n_bad = 0
        self.last_row = ""
        self.seeking_first, self.sought = True, False
        self.done_ids = set()
        self.hw.conveyor(True)
        self._enter("FIRST_MOVE", "Conveyor running", FIRST_MOVE_S)

    def update(self, res):
        """Call once per loop with the newest detection Result (may be None)."""
        if res and res.frame_id != self.tracked_frame:        # keep cocoon ids up to date in every state
            self.tracked_frame = res.frame_id
            self.tracker.update(res, self.hw.conveyor_on)
        now = self.clock()
        el = now - self.t0
        s = self.state

        if s in ("IDLE", "ERROR"):
            if self.hw.start_pressed():
                self.start()
            return
        self.hw.start_event.clear()                   # ignore button presses while a cycle is running

        if s == "FIRST_MOVE":
            if el >= FIRST_MOVE_S:
                self.hw.conveyor(False)
                self._enter("SETTLE", "Settling", SETTLE_S)

        elif s == "ADVANCE":
            if self._fresh(res):
                new = [d for d in zone_dets(res) if d.tid not in self.done_ids]   # not scanned before
                if new and self.first_seen is None:
                    self.first_seen = res.t
                row = find_row(res)
                complete = row is not None and any(d.tid not in self.done_ids for d in row)
                # stop when the new row is complete in the zone - or, if its partner is missing
                # (tilted tray, missed detection), ROW_WAIT_S after its first cocoon arrived
                if complete or (self.first_seen is not None and res.t - self.first_seen >= ROW_WAIT_S):
                    self._stop_and_settle()
                    return
            if el >= DETECTOR_STALL_S and (res is None or now - res.t > DETECTOR_STALL_S):
                self._fail("detector stopped delivering frames")       # never run the belt blind
            elif el >= ADVANCE_TIMEOUT_S:
                if self.rows_done == 0:
                    self._fail("no cocoon reached the stop zone in %.0f s" % ADVANCE_TIMEOUT_S)
                else:
                    self._fail("row %d never arrived (fewer rows than ROWS_PER_TRAY, or row 1 skipped)"
                               % (self.rows_done + 1))

        elif s == "SETTLE":
            if el >= SETTLE_S:
                self.votes, self.misses, self.scan_ids = [], 0, set()
                self._enter("SCAN", "Scanning row %d" % (self.rows_done + 1))

        elif s == "SCAN":
            if self._fresh(res):
                self.scan_ids |= {d.tid for d in zone_dets(res)}
                row = find_row(res)
                if row:
                    self.votes.append((row[0].label, row[1].label))
                else:
                    self.misses += 1
            if len(self.votes) >= SCAN_FRAMES:
                self._decide()
            elif self.misses >= 2 * SCAN_FRAMES or el >= SCAN_TIMEOUT_S:
                self._no_row()

        elif s == "NUDGE":
            if el >= NUDGE_S:
                self.hw.conveyor(False)
                self.hw.motors_forward(self.targets)
                self._enter("PUSH_FWD", "Motor %s forward" % self._names(), PUSH_FORWARD_S)

        elif s == "PUSH_FWD":
            if el >= PUSH_FORWARD_S:
                self.hw.motors_backward(self.targets)
                self._enter("PUSH_BACK", "Motor %s backward" % self._names(), PUSH_BACK_S)

        elif s == "PUSH_BACK":
            if el >= PUSH_BACK_S:
                self.hw.motors_stop()
                self.tracker.reset()      # forget all old cocoon ids, then look at what is really left
                self.scan_ids, self.seen = set(), 0
                self._enter("RECHECK", "Checking the row after the push")

        elif s == "RECHECK":              # belt stopped, pusher retracted: whatever is still in the zone
            if self._fresh(res):          # (pushed-out cocoons that stayed visible) counts as handled
                self.scan_ids |= {d.tid for d in zone_dets(res)}
                self.seen += 1
            if self.seen >= SCAN_FRAMES:
                self.done_ids = set(self.scan_ids)
                self._next_row()
            elif el >= SCAN_TIMEOUT_S:
                self._fail("detector gave no results")

    def _names(self):
        return " + ".join(str(i) for i in self.targets)

    def _no_row(self):
        """The scan could not see a complete row in the stop zone."""
        if not self.misses and not self.votes:
            self._fail("detector gave no results")
        elif self.seeking_first and not self.sought:      # first row may just not have arrived yet:
            self.sought = True                            # creep forward ONCE (never twice - it could
            self._advance("Looking for the first row")    # overshoot, and a second try would skip row 1)
        else:
            self._fail("row %d not found in the stop zone after stopping" % (self.rows_done + 1))

    def _decide(self):
        self.seeking_first = False
        self.done_ids |= self.scan_ids
        left = self._majority([v[0] for v in self.votes])
        right = self._majority([v[1] for v in self.votes])
        self.n_good += (left == GOOD_NAME) + (right == GOOD_NAME)
        self.n_bad += (left != GOOD_NAME) + (right != GOOD_NAME)
        self.targets = ([1] if right != GOOD_NAME else []) + ([2] if left != GOOD_NAME else [])
        self.last_row = "Row %d: LEFT=%s RIGHT=%s" % (self.rows_done + 1, left.upper(), right.upper())
        if not self.targets:
            self._next_row()
        else:
            self.hw.conveyor(True)
            self._enter("NUDGE", "%s -> nudging conveyor" % self.last_row, NUDGE_S)

    @staticmethod
    def _majority(labels):
        """Most common label; a tie counts as bad (safer to reject than to let a defect through)."""
        (top, n), *rest = Counter(labels).most_common()
        return BAD_NAME if rest and rest[0][1] == n else top


# =============================================================================
#  Drawing
# =============================================================================
STATE_COLOR = {"IDLE": GREY, "ERROR": RED, "PUSH_FWD": YELLOW, "PUSH_BACK": YELLOW, "NUDGE": YELLOW}


def draw(frame, res, sorter, hw, det_fps, cam_fps):
    img = frame.copy()
    h, w = img.shape[:2]
    # stop zone
    y1, y2 = int((ROW_CENTER_Y - ROW_TOLERANCE) * h), int((ROW_CENTER_Y + ROW_TOLERANCE) * h)
    cv2.line(img, (0, y1), (w, y1), YELLOW, 1)
    cv2.line(img, (0, y2), (w, y2), YELLOW, 1)
    cv2.putText(img, "STOP ZONE", (w - 105, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
    # detections
    n_zone = 0
    if res:
        n_zone = len(zone_dets(res))
        for d in res.dets:
            col = GREEN if d.label == GOOD_NAME else RED
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
            cv2.putText(img, "%s %.2f" % (d.label.upper(), d.conf), (int(d.x1), max(14, int(d.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    # status panel
    cv2.rectangle(img, (0, 0), (w, 92), (0, 0, 0), -1)
    left_s = sorter.remaining()
    title = sorter.state.replace("_", " ") + ("  %.1fs" % left_s if left_s is not None else "")
    if sorter.state == "ERROR":
        title += "  - press the button to restart"
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATE_COLOR.get(sorter.state, WHITE), 2)
    cv2.putText(img, sorter.msg[:88], (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    running = sorter.state not in ("IDLE", "ERROR")
    cv2.putText(img, "Row %d/%d   Good:%d  Bad:%d   %s" % (min(sorter.rows_done + running, ROWS_PER_TRAY),
                ROWS_PER_TRAY, sorter.n_good, sorter.n_bad, sorter.last_row),
                (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(img, hw.status() + "   in zone:%d   det %.1f fps  cam %.0f fps" % (n_zone, det_fps, cam_fps),
                (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    footer = "S start   X stop   Q quit" + ("    [SIMULATED GPIO - no Raspberry Pi]" if hw.simulated else "")
    cv2.putText(img, footer, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW if hw.simulated else GREY, 1)
    return img


# =============================================================================
#  Main
# =============================================================================
def open_camera(source):
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") and isinstance(source, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # 30 fps on most USB webcams
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Cocoon defect detection and sorting")
    ap.add_argument("--model", default=MODEL_PATH, help="path to best.pt or an NCNN model folder")
    ap.add_argument("--source", default=str(CAMERA_INDEX), help="camera index, or a video file for testing")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if not os.path.exists(args.model):
        sys.exit("Model not found: %r\nCopy the model you trained in Colab (best.pt) next to app.py, "
                 "or pass --model <path>." % args.model)

    hw = Hardware()                                   # relay and motors are OFF from here on
    detector = cap = None
    try:
        if hw.simulated:
            print("No Raspberry Pi GPIO found - running with SIMULATED pins.")
        print("Loading model (the first start takes a while) ...")
        detector = Detector(args.model)
        if set(detector.names.values()) != {GOOD_NAME, BAD_NAME}:
            sys.exit("The model has classes %s but app.py expects exactly '%s' and '%s'.\n"
                     "Fix the class names in Roboflow, or change GOOD_NAME / BAD_NAME at the top of app.py."
                     % (sorted(detector.names.values()), GOOD_NAME, BAD_NAME))
        cap = open_camera(source)
        if not cap.isOpened():
            sys.exit("Cannot open camera / source: %r" % (source,))
        detector.start()
        sorter = Sorter(hw)
        cam_fps, last = 0.0, time.monotonic()
        print("Ready. Press the push button (or S in the window). Q quits.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed - stopping.")
                break
            now = time.monotonic()
            cam_fps = 0.9 * cam_fps + 0.1 / max(now - last, 1e-3)
            last = now
            detector.submit(frame, now)
            if detector.error:
                print("Detector crashed:", detector.error)
                break

            res = detector.result
            sorter.update(res)
            cv2.imshow(WINDOW, draw(frame, res, sorter, hw, detector.fps, cam_fps))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("x"), ord("X"), 27):
                sorter.stop("STOPPED by operator")
            elif key in (ord("s"), ord("S")) and sorter.state in ("IDLE", "ERROR"):
                hw.start_event.set()
    except KeyboardInterrupt:
        pass
    finally:
        hw.all_off()                                  # never leave the conveyor or motors running
        if detector:
            detector.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        hw.close()
        print("Stopped. All outputs off.")


if __name__ == "__main__":
    main()
