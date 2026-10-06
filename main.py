
"""
===============================================================
NEXUS — NORMAL MODE v3.1
===============================================================

GESTURES
---------------------------------------------------------------

☝️ INDEX ONLY
    -> Cursor

🤏 THUMB + INDEX QUICK PINCH
    -> ONE CLICK

🤏 THUMB + INDEX HOLD
    -> DRAG / SELECT

✌️ INDEX + MIDDLE
    -> SCROLL

✋ OPEN PALM HOLD
    -> PLAY / PAUSE

🤙 SHAKA
    -> MUTE / UNMUTE

✊ FIST + MOVE UP / DOWN
    -> VOLUME UP / DOWN

👐 TWO HANDS
    -> ZOOM

ZOOM:
    Each hand's thumb-index distance is measured separately.

    Both pairs moving APART
        -> ZOOM IN

    Both pairs moving TOGETHER
        -> ZOOM OUT

    Distance between the two hands
        -> IGNORED

REMOVED
---------------------------------------------------------------
❌ Next track
❌ Previous track
❌ Fist swipe
❌ Fist mute

Keyboard mode will be added later.
===============================================================
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ===============================================================
# OPTIONAL WINDOWS VOLUME BACKEND
# ===============================================================

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import (
        AudioUtilities,
        IAudioEndpointVolume
    )

    PYCAW_AVAILABLE = True

except ImportError:
    PYCAW_AVAILABLE = False


# ===============================================================
# CONFIG
# ===============================================================

MODEL_PATH = "hand_landmarker.task"

CAM_W = 480
CAM_H = 360

DEBUG_OVERLAY = True


# ===============================================================
# CAMERA / CURSOR
# ===============================================================

FRAME_MARGIN_X = 50
FRAME_MARGIN_Y = 40

# Higher = more responsive.
# Lower = smoother but more laggy.
CURSOR_SMOOTHING = 0.55


# ===============================================================
# CLICK / DRAG
# ===============================================================

PINCH_ENTER_RATIO = 0.35
PINCH_EXIT_RATIO = 0.55

DRAG_HOLD_TIME = 0.30

CLICK_COOLDOWN = 0.35


# ===============================================================
# SCROLL
# ===============================================================

SCROLL_DEADZONE = 2.0
SCROLL_SENSITIVITY = 3


# ===============================================================
# PLAY / PAUSE
# ===============================================================

PALM_HOLD_TIME = 0.60
PLAYPAUSE_COOLDOWN = 0.80


# ===============================================================
# MUTE
# ===============================================================

MUTE_COOLDOWN = 0.80


# ===============================================================
# VOLUME
# ===============================================================

VOLUME_DEADZONE = 1.5
VOLUME_SENSITIVITY = 0.55
VOLUME_UPDATE_INTERVAL = 0.04


# ===============================================================
# ZOOM
# ===============================================================

# Tiny changes below this are treated as camera noise.
ZOOM_DEADZONE = 0.008

# Smaller = faster zoom response.
ZOOM_STEP = 0.012

# Prevents a crazy number of zoom events in one frame.
ZOOM_MAX_STEPS = 4

ZOOM_COOLDOWN = 0.05


# ===============================================================
# PYAUTOGUI
# ===============================================================

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()


# ===============================================================
# HAND CONNECTIONS
# ===============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ===============================================================
# VOLUME BACKEND
# ===============================================================

class VolumeBackend:

    def __init__(self):

        self.iface = None
        self.last_level = 50.0

        if PYCAW_AVAILABLE:

            try:

                devices = AudioUtilities.GetSpeakers()

                interface = devices.Activate(
                    IAudioEndpointVolume._iid_,
                    CLSCTX_ALL,
                    None
                )

                self.iface = cast(
                    interface,
                    POINTER(IAudioEndpointVolume)
                )

                self.last_level = (
                    self.iface.GetMasterVolumeLevelScalar()
                    * 100.0
                )

            except Exception as e:

                print(
                    "PyCaw initialization failed:",
                    e
                )

                self.iface = None


    def get_volume(self):

        if self.iface is not None:

            try:

                return (
                    self.iface.GetMasterVolumeLevelScalar()
                    * 100.0
                )

            except Exception:

                pass

        return self.last_level


    def set_volume(self, pct):

        pct = max(
            0.0,
            min(
                100.0,
                float(pct)
            )
        )

        if self.iface is not None:

            try:

                self.iface.SetMasterVolumeLevelScalar(
                    pct / 100.0,
                    None
                )

                self.last_level = pct
                return

            except Exception:

                pass

        # Fallback.
        current = self.last_level

        if pct > current + 0.5:

            pyautogui.press("volumeup")

            self.last_level = min(
                100.0,
                current + 2
            )

        elif pct < current - 0.5:

            pyautogui.press("volumedown")

            self.last_level = max(
                0.0,
                current - 2
            )


    def change_volume(self, delta):

        current = self.get_volume()

        self.set_volume(
            current + delta
        )


    def toggle_mute(self):

        if self.iface is not None:

            try:

                muted = self.iface.GetMute()

                self.iface.SetMute(
                    0 if muted else 1,
                    None
                )

                return

            except Exception:

                pass

        pyautogui.press("volumemute")


# ===============================================================
# GEOMETRY
# ===============================================================

def dist(
    p1,
    p2,
    w,
    h
):

    return math.hypot(
        (p1.x - p2.x) * w,
        (p1.y - p2.y) * h
    )


def palm_size(
    landmarks,
    w,
    h
):

    return max(
        dist(
            landmarks[0],
            landmarks[9],
            w,
            h
        ),
        1.0
    )


# ===============================================================
# THUMB DETECTION
# ===============================================================

def thumb_extended(
    landmarks,
    w,
    h
):

    ref = dist(
        landmarks[2],
        landmarks[17],
        w,
        h
    )

    tip = dist(
        landmarks[4],
        landmarks[17],
        w,
        h
    )

    return (
        tip
        >
        ref * 1.15
    )


# ===============================================================
# FINGER DETECTION
# ===============================================================

def fingers_up(
    landmarks,
    w,
    h
):

    fingers = []

    # Thumb.
    fingers.append(
        1
        if thumb_extended(
            landmarks,
            w,
            h
        )
        else 0
    )

    # Index, middle, ring, pinky.
    for tip in [8, 12, 16, 20]:

        pip = tip - 2

        fingers.append(
            1
            if landmarks[tip].y
            <
            landmarks[pip].y
            else 0
        )

    return fingers


# ===============================================================
# DRAW HAND
# ===============================================================

def draw_hand(
    frame,
    landmarks,
    w,
    h
):

    pts = [
        (
            int(lm.x * w),
            int(lm.y * h)
        )
        for lm in landmarks
    ]

    for a, b in HAND_CONNECTIONS:

        cv2.line(
            frame,
            pts[a],
            pts[b],
            (0, 255, 120),
            2
        )

    for p in pts:

        cv2.circle(
            frame,
            p,
            4,
            (0, 0, 255),
            -1
        )


# ===============================================================
# GESTURE CONTROLLER
# ===============================================================

class GestureController:

    def __init__(self):

        # -------------------------------------------------------
        # CURSOR
        # -------------------------------------------------------

        self.prev_cursor = None


        # -------------------------------------------------------
        # CLICK / DRAG
        # -------------------------------------------------------

        self.pinched = False
        self.pinch_start_time = None

        self.dragging = False

        self.last_click_time = 0


        # -------------------------------------------------------
        # SCROLL
        # -------------------------------------------------------

        self.prev_scroll_y = None


        # -------------------------------------------------------
        # VOLUME
        # -------------------------------------------------------

        self.prev_volume_y = None
        self.last_volume_update = 0

        self.volume = VolumeBackend()


        # -------------------------------------------------------
        # PLAY / PAUSE
        # -------------------------------------------------------

        self.palm_hold_start = None
        self.palm_triggered = False

        self.last_playpause_time = 0


        # -------------------------------------------------------
        # SHAKA
        # -------------------------------------------------------

        self.shaka_active = False
        self.last_mute_time = 0


        # -------------------------------------------------------
        # ZOOM
        # -------------------------------------------------------

        self.prev_zoom_ratio = None
        self.zoom_accumulator = 0.0
        self.last_zoom_time = 0


        # -------------------------------------------------------
        # UI
        # -------------------------------------------------------

        self.status_text = "IDLE"
        self.debug_text = ""


    # ===========================================================
    # CURSOR
    # ===========================================================

    def handle_cursor(
        self,
        landmarks,
        w,
        h
    ):

        index_tip = landmarks[8]

        raw_x = index_tip.x * w
        raw_y = index_tip.y * h


        x = np.interp(
            raw_x,
            (
                FRAME_MARGIN_X,
                w - FRAME_MARGIN_X
            ),
            (
                0,
                SCREEN_W
            )
        )

        y = np.interp(
            raw_y,
            (
                FRAME_MARGIN_Y,
                h - FRAME_MARGIN_Y
            ),
            (
                0,
                SCREEN_H
            )
        )


        x = np.clip(
            x,
            0,
            SCREEN_W - 1
        )

        y = np.clip(
            y,
            0,
            SCREEN_H - 1
        )


        if self.prev_cursor is None:

            cx = x
            cy = y

        else:

            px, py = self.prev_cursor

            cx = (
                px
                +
                (x - px)
                * CURSOR_SMOOTHING
            )

            cy = (
                py
                +
                (y - py)
                * CURSOR_SMOOTHING
            )


        pyautogui.moveTo(
            int(cx),
            int(cy),
            _pause=False
        )

        self.prev_cursor = (
            cx,
            cy
        )

        self.status_text = "CURSOR"


    # ===========================================================
    # CLICK / DRAG
    # ===========================================================

    def handle_pinch(
        self,
        landmarks,
        w,
        h
    ):

        ps = palm_size(
            landmarks,
            w,
            h
        )

        d = dist(
            landmarks[4],
            landmarks[8],
            w,
            h
        )

        enter_threshold = (
            ps
            *
            PINCH_ENTER_RATIO
        )

        exit_threshold = (
            ps
            *
            PINCH_EXIT_RATIO
        )

        now = time.time()


        self.debug_text = (
            f"PINCH "
            f"d={d:.1f} "
            f"enter={enter_threshold:.1f} "
            f"exit={exit_threshold:.1f}"
        )


        # -------------------------------------------------------
        # PINCH START
        # -------------------------------------------------------

        if (
            not self.pinched
            and
            d < enter_threshold
        ):

            self.pinched = True

            self.pinch_start_time = now

            self.status_text = "PINCH"

            return True


        # -------------------------------------------------------
        # PINCH ACTIVE
        # -------------------------------------------------------

        if self.pinched:

            held = (
                now
                -
                (
                    self.pinch_start_time
                    if self.pinch_start_time
                    else now
                )
            )


            # ---------------------------------------------------
            # DRAG
            # ---------------------------------------------------

            if held >= DRAG_HOLD_TIME:

                # Keep moving cursor while dragging.
                self.handle_cursor(
                    landmarks,
                    w,
                    h
                )


                if not self.dragging:

                    pyautogui.mouseDown()

                    self.dragging = True


                self.status_text = (
                    "DRAG / SELECT"
                )


            else:

                self.status_text = "PINCH"


            # ---------------------------------------------------
            # RELEASE
            # ---------------------------------------------------

            if d > exit_threshold:

                self.pinched = False


                if self.dragging:

                    pyautogui.mouseUp()

                    self.dragging = False

                    self.status_text = (
                        "DRAG RELEASE"
                    )

                else:

                    if (
                        now
                        -
                        self.last_click_time
                        >
                        CLICK_COOLDOWN
                    ):

                        pyautogui.click()

                        self.last_click_time = now

                        self.status_text = "CLICK"


                self.pinch_start_time = None


            return True


        return False


    # ===========================================================
    # SCROLL
    # ===========================================================

    def handle_scroll(
        self,
        landmarks,
        w,
        h
    ):

        avg_y = (
            landmarks[8].y
            +
            landmarks[12].y
        ) / 2 * h


        if self.prev_scroll_y is not None:

            delta = (
                self.prev_scroll_y
                -
                avg_y
            )

            if abs(delta) > SCROLL_DEADZONE:

                amount = int(
                    delta
                    *
                    SCROLL_SENSITIVITY
                )

                if amount != 0:

                    pyautogui.scroll(
                        amount
                    )

                    self.status_text = "SCROLL"


        self.prev_scroll_y = avg_y


    # ===========================================================
    # VOLUME
    # ===========================================================

    def handle_volume(
        self,
        landmarks,
        w,
        h
    ):

        hand_y = (
            landmarks[0].y
            +
            landmarks[9].y
        ) / 2 * h


        now = time.time()


        if self.prev_volume_y is None:

            self.prev_volume_y = hand_y

            current = int(
                self.volume.get_volume()
            )

            self.status_text = (
                f"VOLUME {current}%"
            )

            return


        delta_y = (
            self.prev_volume_y
            -
            hand_y
        )


        self.prev_volume_y = hand_y


        if abs(delta_y) < VOLUME_DEADZONE:

            current = int(
                self.volume.get_volume()
            )

            self.status_text = (
                f"VOLUME {current}%"
            )

            return


        if (
            now
            -
            self.last_volume_update
            <
            VOLUME_UPDATE_INTERVAL
        ):

            return


        self.last_volume_update = now


        change = (
            delta_y
            *
            VOLUME_SENSITIVITY
        )


        self.volume.change_volume(
            change
        )


        current = int(
            self.volume.get_volume()
        )


        if change > 0:

            self.status_text = (
                f"VOLUME ↑ {current}%"
            )

        else:

            self.status_text = (
                f"VOLUME ↓ {current}%"
            )


    # ===========================================================
    # PLAY / PAUSE
    # ===========================================================

    def handle_playpause(self):

        now = time.time()


        if self.palm_hold_start is None:

            self.palm_hold_start = now

            self.palm_triggered = False


        held = (
            now
            -
            self.palm_hold_start
        )


        if (
            held >= PALM_HOLD_TIME
            and
            not self.palm_triggered
            and
            now
            -
            self.last_playpause_time
            >
            PLAYPAUSE_COOLDOWN
        ):

            pyautogui.press(
                "space"
            )

            self.palm_triggered = True

            self.last_playpause_time = now

            self.status_text = (
                "PLAY / PAUSE"
            )

        else:

            self.status_text = (
                f"PALM {held:.1f}s"
            )


    # ===========================================================
    # SHAKA = MUTE / UNMUTE
    # ===========================================================

    def handle_shaka(self):

        now = time.time()


        # Trigger once per shaka.
        # Release -> shaka again -> trigger again.

        if not self.shaka_active:

            if (
                now
                -
                self.last_mute_time
                >
                MUTE_COOLDOWN
            ):

                self.volume.toggle_mute()

                self.last_mute_time = now

                self.status_text = (
                    "MUTE / UNMUTE"
                )

            self.shaka_active = True

        else:

            self.status_text = (
                "SHAKA — RELEASE"
            )


    # ===========================================================
    # TWO-HAND ZOOM
    # ===========================================================

    def handle_zoom(
        self,
        lm1,
        lm2,
        w,
        h
    ):
        """
        TWO-HAND ZOOM

        IMPORTANT:

        We DO NOT measure distance between the hands.

        We measure:

            LEFT  thumb <-> index
            RIGHT thumb <-> index

        independently.

        Both distances increasing:
            -> ZOOM IN

        Both distances decreasing:
            -> ZOOM OUT
        """

        # -------------------------------------------------------
        # THUMB <-> INDEX DISTANCE
        # -------------------------------------------------------

        d1 = dist(
            lm1[4],
            lm1[8],
            w,
            h
        )

        d2 = dist(
            lm2[4],
            lm2[8],
            w,
            h
        )


        # -------------------------------------------------------
        # NORMALIZE EACH HAND
        # -------------------------------------------------------

        palm1 = palm_size(
            lm1,
            w,
            h
        )

        palm2 = palm_size(
            lm2,
            w,
            h
        )


        ratio1 = d1 / palm1
        ratio2 = d2 / palm2


        # Average.
        zoom_value = (
            ratio1
            +
            ratio2
        ) / 2.0


        # -------------------------------------------------------
        # FIRST FRAME
        # -------------------------------------------------------

        if self.prev_zoom_ratio is None:

            self.prev_zoom_ratio = zoom_value

            self.zoom_accumulator = 0.0

            self.status_text = (
                "ZOOM READY"
            )

            self.debug_text = (
                f"ZOOM "
                f"L={ratio1:.2f} "
                f"R={ratio2:.2f}"
            )

            return


        # -------------------------------------------------------
        # CHANGE
        # -------------------------------------------------------

        delta = (
            zoom_value
            -
            self.prev_zoom_ratio
        )


        self.prev_zoom_ratio = zoom_value


        self.debug_text = (
            f"ZOOM "
            f"L={ratio1:.2f} "
            f"R={ratio2:.2f} "
            f"D={delta:+.3f}"
        )


        # -------------------------------------------------------
        # IGNORE TINY CAMERA MOVEMENT
        # -------------------------------------------------------

        if abs(delta) < ZOOM_DEADZONE:

            self.status_text = (
                "ZOOM READY"
            )

            return


        # -------------------------------------------------------
        # ACCUMULATE
        # -------------------------------------------------------

        self.zoom_accumulator += delta


        # -------------------------------------------------------
        # COOLDOWN
        # -------------------------------------------------------

        now = time.time()


        if (
            now
            -
            self.last_zoom_time
            <
            ZOOM_COOLDOWN
        ):

            return


        # -------------------------------------------------------
        # NOT ENOUGH MOVEMENT YET
        # -------------------------------------------------------

        if (
            abs(self.zoom_accumulator)
            <
            ZOOM_STEP
        ):

            self.status_text = (
                "ZOOM READY"
            )

            return


        # -------------------------------------------------------
        # DIRECTION
        # -------------------------------------------------------

        if self.zoom_accumulator > 0:

            direction = 1

        else:

            direction = -1


        # -------------------------------------------------------
        # NUMBER OF STEPS
        # -------------------------------------------------------

        steps = int(
            abs(
                self.zoom_accumulator
            )
            /
            ZOOM_STEP
        )


        steps = max(
            1,
            min(
                ZOOM_MAX_STEPS,
                steps
            )
        )


        # Consume movement.
        self.zoom_accumulator -= (
            direction
            *
            steps
            *
            ZOOM_STEP
        )


        # -------------------------------------------------------
        # ACTUAL ZOOM
        # -------------------------------------------------------

        for _ in range(steps):

            pyautogui.keyDown(
                "ctrl"
            )


            if direction > 0:

                # Ctrl + +
                pyautogui.keyDown(
                    "shift"
                )

                pyautogui.press(
                    "="
                )

                pyautogui.keyUp(
                    "shift"
                )

            else:

                # Ctrl + -
                pyautogui.press(
                    "-"
                )


            pyautogui.keyUp(
                "ctrl"
            )


        self.last_zoom_time = now


        # -------------------------------------------------------
        # STATUS
        # -------------------------------------------------------

        if direction > 0:

            self.status_text = (
                f"ZOOM IN ↑ x{steps}"
            )

        else:

            self.status_text = (
                f"ZOOM OUT ↓ x{steps}"
            )


    # ===========================================================
    # RESET STATES
    # ===========================================================

    def reset_states_except(
        self,
        active
    ):

        # -------------------------------------------------------
        # PINCH / DRAG
        # -------------------------------------------------------

        if active != "pinch":

            if self.dragging:

                pyautogui.mouseUp()

                self.dragging = False


            self.pinched = False

            self.pinch_start_time = None


        # -------------------------------------------------------
        # SCROLL
        # -------------------------------------------------------

        if active != "scroll":

            self.prev_scroll_y = None


        # -------------------------------------------------------
        # VOLUME
        # -------------------------------------------------------

        if active != "volume":

            self.prev_volume_y = None

            self.last_volume_update = 0


        # -------------------------------------------------------
        # PALM
        # -------------------------------------------------------

        if active != "palm":

            self.palm_hold_start = None

            self.palm_triggered = False


        # -------------------------------------------------------
        # SHAKA
        # -------------------------------------------------------

        if active != "shaka":

            self.shaka_active = False


        # -------------------------------------------------------
        # CURSOR
        # -------------------------------------------------------

        if active != "cursor":

            self.prev_cursor = None


        # -------------------------------------------------------
        # ZOOM
        # -------------------------------------------------------

        if active != "zoom":

            self.prev_zoom_ratio = None

            self.zoom_accumulator = 0.0


# ===============================================================
# MAIN
# ===============================================================

def main():

    # ===========================================================
    # MEDIAPIPE
    # ===========================================================

    base_options = (
        mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        )
    )


    options = (
        vision.HandLandmarkerOptions(

            base_options=base_options,

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_hands=2,

            min_hand_detection_confidence=0.6,

            min_hand_presence_confidence=0.6,

            min_tracking_confidence=0.6
        )
    )


    landmarker = (
        vision.HandLandmarker
        .create_from_options(
            options
        )
    )


    controller = (
        GestureController()
    )


    # ===========================================================
    # CAMERA
    # ===========================================================

    cap = cv2.VideoCapture(0)

    # Reduce camera buffering.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAM_W
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAM_H
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        60
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        landmarker.close()

        return


    # ===========================================================
    # MAIN LOOP
    # ===========================================================

    while cap.isOpened():

        ok, frame = cap.read()


        if not ok:

            print(
                "Camera read failed."
            )

            break


        # Mirror camera.
        frame = cv2.flip(
            frame,
            1
        )


        h, w = (
            frame.shape[:2]
        )


        # -------------------------------------------------------
        # RGB
        # -------------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )


        # -------------------------------------------------------
        # REAL TIMESTAMP
        # -------------------------------------------------------

        timestamp_ms = int(
            time.time() * 1000
        )


        # -------------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------------

        result = (
            landmarker
            .detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        active_gesture = "none"


        # =======================================================
        # HANDS FOUND
        # =======================================================

        if result.hand_landmarks:

            hands_data = []


            for i, landmarks in enumerate(
                result.hand_landmarks
            ):

                label = (
                    result
                    .handedness[i][0]
                    .category_name
                )


                draw_hand(
                    frame,
                    landmarks,
                    w,
                    h
                )


                hands_data.append(
                    (
                        landmarks,
                        label
                    )
                )


            # ===================================================
            # TWO HANDS = ZOOM
            # ===================================================

            if len(hands_data) == 2:

                lm1, label1 = (
                    hands_data[0]
                )

                lm2, label2 = (
                    hands_data[1]
                )


                controller.handle_zoom(
                    lm1,
                    lm2,
                    w,
                    h
                )


                active_gesture = "zoom"


            # ===================================================
            # ONE HAND
            # ===================================================

            else:

                landmarks, label = (
                    hands_data[0]
                )


                # ------------------------------------------------
                # PINCH HAS HIGHEST PRIORITY
                # ------------------------------------------------

                pinch_active = (
                    controller.handle_pinch(
                        landmarks,
                        w,
                        h
                    )
                )


                if pinch_active:

                    active_gesture = "pinch"


                else:

                    f = fingers_up(
                        landmarks,
                        w,
                        h
                    )


                    # ============================================
                    # SHAKA
                    # ============================================

                    if (
                        f[0] == 1
                        and
                        f[1] == 0
                        and
                        f[2] == 0
                        and
                        f[3] == 0
                        and
                        f[4] == 1
                    ):

                        active_gesture = "shaka"

                        controller.handle_shaka()


                    # ============================================
                    # SCROLL
                    # ============================================

                    elif (
                        f[1] == 1
                        and
                        f[2] == 1
                        and
                        f[3] == 0
                        and
                        f[4] == 0
                    ):

                        active_gesture = "scroll"

                        controller.handle_scroll(
                            landmarks,
                            w,
                            h
                        )


                    # ============================================
                    # OPEN PALM
                    # ============================================

                    elif (
                        f[1] == 1
                        and
                        f[2] == 1
                        and
                        f[3] == 1
                        and
                        f[4] == 1
                    ):

                        active_gesture = "palm"

                        controller.handle_playpause()


                    # ============================================
                    # FIST = VOLUME
                    # ============================================

                    elif (
                        f[1] == 0
                        and
                        f[2] == 0
                        and
                        f[3] == 0
                        and
                        f[4] == 0
                    ):

                        active_gesture = "volume"

                        controller.handle_volume(
                            landmarks,
                            w,
                            h
                        )


                    # ============================================
                    # INDEX ONLY = CURSOR
                    # ============================================

                    elif (
                        f[1] == 1
                        and
                        f[2] == 0
                        and
                        f[3] == 0
                        and
                        f[4] == 0
                    ):

                        active_gesture = "cursor"

                        controller.handle_cursor(
                            landmarks,
                            w,
                            h
                        )


        # =======================================================
        # RESET STATES
        # =======================================================

        controller.reset_states_except(
            active_gesture
        )


        # =======================================================
        # UI
        # =======================================================

        cv2.putText(
            frame,
            "NEXUS | NORMAL MODE",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Action: {controller.status_text}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2
        )


        if DEBUG_OVERLAY:

            cv2.putText(
                frame,
                controller.debug_text,
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 200, 255),
                1
            )


        cv2.putText(
            frame,
            "Q = Quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )


        cv2.imshow(
            "NEXUS - Normal Mode",
            frame
        )


        # =======================================================
        # QUIT
        # =======================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


    # ===========================================================
    # CLEANUP
    # ===========================================================

    if controller.dragging:

        pyautogui.mouseUp()


    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


# ===============================================================
# START
# ===============================================================

if __name__ == "__main__":

    main()
