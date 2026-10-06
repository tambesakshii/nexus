"""
NEXUS — Virtual Keyboard v4
============================

KEYBOARD STATES
---------------

UNLOCKED:
    Two index fingers apart  -> show keyboard
    Two index fingers together -> hide keyboard

LOCKED:
    Two-index activation gesture is DISABLED.
    Keyboard stays visible.
    Both hands are free for keyboard interaction.

LOCK / UNLOCK:
    Button is OUTSIDE the keyboard.
    It can be activated with the normal NEXUS click system.

Typing/pinch is intentionally NOT implemented here yet.
That comes next.
"""

import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ==============================================================
# CONFIG
# ==============================================================

MODEL_PATH = "hand_landmarker.task"

CAM_W = 640
CAM_H = 480

DEBUG = True

# Baby-pink glass opacity
KEYBOARD_OPACITY = 0.25

# --------------------------------------------------------------
# Keyboard activation
# --------------------------------------------------------------

OPEN_THRESHOLD = 2.0
CLOSE_THRESHOLD = 1.55

DISTANCE_SMOOTHING = 0.40

# --------------------------------------------------------------
# Virtual pointer
# --------------------------------------------------------------

CURSOR_SMOOTHING = 0.35

# --------------------------------------------------------------
# Lock button
# --------------------------------------------------------------

LOCK_BUTTON_WIDTH = 48
LOCK_BUTTON_HEIGHT = 28


# ==============================================================
# HAND CONNECTIONS
# ==============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


# ==============================================================
# GEOMETRY
# ==============================================================

def distance(p1, p2, w, h):

    return math.hypot(
        (p1.x - p2.x) * w,
        (p1.y - p2.y) * h
    )


def palm_size(landmarks, w, h):

    return max(
        distance(
            landmarks[0],
            landmarks[9],
            w,
            h
        ),
        1.0
    )


# ==============================================================
# DRAW HAND
# ==============================================================

def draw_hand(frame, landmarks, w, h):

    points = [
        (
            int(lm.x * w),
            int(lm.y * h)
        )
        for lm in landmarks
    ]

    for a, b in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 180, 210),
            2
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (255, 105, 160),
            -1
        )

    # Index fingertip
    cv2.circle(
        frame,
        points[8],
        9,
        (255, 105, 180),
        2
    )


# ==============================================================
# ROUNDED RECTANGLE
# ==============================================================

def rounded_rectangle(
    image,
    top_left,
    bottom_right,
    radius,
    fill,
    border=None,
    border_thickness=1
):

    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(
        image,
        (x1 + radius, y1),
        (x2 - radius, y2),
        fill,
        -1
    )

    cv2.rectangle(
        image,
        (x1, y1 + radius),
        (x2, y2 - radius),
        fill,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y1 + radius),
        radius,
        fill,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y1 + radius),
        radius,
        fill,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y2 - radius),
        radius,
        fill,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y2 - radius),
        radius,
        fill,
        -1
    )

    if border is not None:

        cv2.line(
            image,
            (x1 + radius, y1),
            (x2 - radius, y1),
            border,
            border_thickness
        )

        cv2.line(
            image,
            (x1 + radius, y2),
            (x2 - radius, y2),
            border,
            border_thickness
        )

        cv2.line(
            image,
            (x1, y1 + radius),
            (x1, y2 - radius),
            border,
            border_thickness
        )

        cv2.line(
            image,
            (x2, y1 + radius),
            (x2, y2 - radius),
            border,
            border_thickness
        )


# ==============================================================
# KEYBOARD LAYOUT
# ==============================================================

KEYBOARD_ROWS = [

    [
        ("`", 1),
        ("1", 1),
        ("2", 1),
        ("3", 1),
        ("4", 1),
        ("5", 1),
        ("6", 1),
        ("7", 1),
        ("8", 1),
        ("9", 1),
        ("0", 1),
        ("-", 1),
        ("=", 1),
        ("BACKSPACE", 2.0)
    ],

    [
        ("TAB", 1.4),
        ("Q", 1),
        ("W", 1),
        ("E", 1),
        ("R", 1),
        ("T", 1),
        ("Y", 1),
        ("U", 1),
        ("I", 1),
        ("O", 1),
        ("P", 1),
        ("[", 1),
        ("]", 1),
        ("\\", 1.4)
    ],

    [
        ("CAPS", 1.7),
        ("A", 1),
        ("S", 1),
        ("D", 1),
        ("F", 1),
        ("G", 1),
        ("H", 1),
        ("J", 1),
        ("K", 1),
        ("L", 1),
        (";", 1),
        ("'", 1),
        ("ENTER", 1.8)
    ],

    [
        ("SHIFT", 2.0),
        ("Z", 1),
        ("X", 1),
        ("C", 1),
        ("V", 1),
        ("B", 1),
        ("N", 1),
        ("M", 1),
        (",", 1),
        (".", 1),
        ("/", 1),
        ("SHIFT", 2.0)
    ],

    [
        ("CTRL", 1.3),
        ("ALT", 1.3),
        ("FN", 1.2),
        ("SPACE", 5.5),
        ("ALT", 1.3),
        ("←", 1),
        ("↓", 1),
        ("↑", 1),
        ("→", 1)
    ]
]


# ==============================================================
# KEYBOARD GEOMETRY
# ==============================================================

def get_keyboard_geometry(frame):

    h, w = frame.shape[:2]

    keyboard_width = min(
        620,
        w - 18
    )

    keyboard_height = 315

    x1 = (
        w - keyboard_width
    ) // 2

    y1 = (
        h
        - keyboard_height
        - 8
    )

    x2 = x1 + keyboard_width
    y2 = y1 + keyboard_height

    gap = 4

    usable_width = keyboard_width - 18

    max_units = max(
        sum(
            unit
            for _, unit in row
        )
        for row in KEYBOARD_ROWS
    )

    max_keys = max(
        len(row)
        for row in KEYBOARD_ROWS
    )

    base_key_width = (
        usable_width
        -
        gap * (max_keys - 1)
    ) / max_units

    key_height = 40

    start_y = y1 + 28

    keys = []

    for row_index, row in enumerate(KEYBOARD_ROWS):

        total_units = sum(
            unit
            for _, unit in row
        )

        row_width = (
            total_units * base_key_width
            +
            (len(row) - 1) * gap
        )

        row_x = (
            x1
            +
            (
                keyboard_width
                -
                row_width
            )
            / 2
        )

        current_x = row_x

        y = (
            start_y
            +
            row_index
            *
            (key_height + gap)
        )

        for key_text, units in row:

            current_width = (
                base_key_width * units
                +
                gap * (units - 1)
            )

            kx1 = int(current_x)
            ky1 = int(y)

            kx2 = int(
                current_x
                +
                current_width
            )

            ky2 = int(
                y
                +
                key_height
            )

            keys.append({
                "text": key_text,
                "x1": kx1,
                "y1": ky1,
                "x2": kx2,
                "y2": ky2
            })

            current_x += (
                current_width
                +
                gap
            )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "keys": keys
    }


# ==============================================================
# FIND KEY
# ==============================================================

def get_hovered_key(x, y, geometry):

    for key in geometry["keys"]:

        if (
            key["x1"] <= x <= key["x2"]
            and
            key["y1"] <= y <= key["y2"]
        ):

            return key

    return None


# ==============================================================
# LOCK BUTTON GEOMETRY
# ==============================================================

def get_lock_button_geometry(frame):

    h, w = frame.shape[:2]

    x2 = w - 12
    x1 = x2 - LOCK_BUTTON_WIDTH

    y1 = 12
    y2 = y1 + LOCK_BUTTON_HEIGHT

    return x1, y1, x2, y2


# ==============================================================
# DRAW LOCK BUTTON
# ==============================================================

def draw_lock_button(frame, locked, hovered=False):
    x1, y1, x2, y2 = get_lock_button_geometry(frame)
    fill = (255, 145, 180) if hovered else (255, 205, 220)
    border = (255, 80, 145) if hovered else (235, 145, 175)
    text_color = (255, 255, 255) if hovered else (120, 55, 80)
    rounded_rectangle(frame, (x1, y1), (x2, y2), 7, fill, border, 1)
    text = 'LOCK' if locked else 'UNLK'
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0]
    tx = x1 + (x2-x1-size[0])//2
    ty = y1 + (y2-y1+size[1])//2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, text_color, 1, cv2.LINE_AA)

def is_inside_lock_button(x, y, frame):
    x1, y1, x2, y2 = get_lock_button_geometry(frame)
    return x1 <= x <= x2 and y1 <= y <= y2


# ==============================================================
# DRAW KEYBOARD
# ==============================================================

def draw_keyboard(
    frame,
    hovered_keys
):

    geometry = get_keyboard_geometry(frame)

    x1 = geometry["x1"]
    y1 = geometry["y1"]
    x2 = geometry["x2"]
    y2 = geometry["y2"]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (255, 185, 205),
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        KEYBOARD_OPACITY,
        frame,
        1.0 - KEYBOARD_OPACITY,
        0
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 105, 160),
        2
    )

    # Header

    cv2.putText(
        frame,
        "NEXUS",
        (x1 + 12, y1 + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 80, 145),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "VIRTUAL KEYBOARD",
        (x1 + 70, y1 + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (120, 70, 90),
        1,
        cv2.LINE_AA
    )


    # Keys

    for key in geometry["keys"]:

        kx1 = key["x1"]
        ky1 = key["y1"]
        kx2 = key["x2"]
        ky2 = key["y2"]

        is_hovered = any(
            hovered is not None
            and hovered["x1"] == kx1
            and hovered["y1"] == ky1
            for hovered in hovered_keys
        )


        # Shadow

        rounded_rectangle(
            frame,
            (
                kx1 + 2,
                ky1 + 3
            ),
            (
                kx2 + 2,
                ky2 + 3
            ),
            6,
            (220, 140, 165)
        )


        # Key body

        if is_hovered:

            rounded_rectangle(
                frame,
                (kx1, ky1),
                (kx2, ky2),
                6,
                (255, 145, 180),
                (255, 80, 145),
                2
            )

        else:

            rounded_rectangle(
                frame,
                (kx1, ky1),
                (kx2, ky2),
                6,
                (255, 210, 222),
                (235, 145, 175),
                1
            )


        # Shine

        cv2.line(
            frame,
            (
                kx1 + 7,
                ky1 + 3
            ),
            (
                kx2 - 7,
                ky1 + 3
            ),
            (255, 235, 242),
            1
        )


        # Font

        if len(key["text"]) >= 8:

            font_scale = 0.23

        elif len(key["text"]) >= 5:

            font_scale = 0.27

        elif len(key["text"]) >= 3:

            font_scale = 0.31

        else:

            font_scale = 0.43


        text_size = cv2.getTextSize(
            key["text"],
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1
        )[0]

        tx = (
            kx1
            +
            (
                kx2
                -
                kx1
                -
                text_size[0]
            )
            // 2
        )

        ty = (
            ky1
            +
            (
                ky2
                -
                ky1
                +
                text_size[1]
            )
            // 2
        )


        if is_hovered:

            text_color = (
                255,
                255,
                255
            )

        else:

            text_color = (
                120,
                55,
                80
            )


        cv2.putText(
            frame,
            key["text"],
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA
        )


    return geometry


# ==============================================================
# MAIN
# ==============================================================

def main():

    # ==========================================================
    # MEDIAPIPE
    # ==========================================================

    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    landmarker = (
        vision.HandLandmarker
        .create_from_options(options)
    )


    # ==========================================================
    # CAMERA
    # ==========================================================

    cap = cv2.VideoCapture(0)

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

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        landmarker.close()

        return


    # ==========================================================
    # STATE
    # ==========================================================

    keyboard_visible = False

    keyboard_locked = False

    smoothed_distance = None

    hovered_key = None

    lock_hovered = False

    cursor_x = None
    cursor_y = None

    frame_timestamp_ms = 0

    lock_pinch_active = False
    lock_last_click = 0.0
    LOCK_PINCH_ENTER_RATIO = 0.35
    LOCK_PINCH_EXIT_RATIO = 0.55
    LOCK_CLICK_COOLDOWN = 0.45


    # ==========================================================
    # LOOP
    # ==========================================================

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:

            print(
                "Camera read failed."
            )

            break


        frame = cv2.flip(
            frame,
            1
        )

        h, w = frame.shape[:2]


        # ======================================================
        # MEDIAPIPE
        # ======================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_timestamp_ms += 33

        result = (
            landmarker
            .detect_for_video(
                mp_image,
                frame_timestamp_ms
            )
        )


        # ======================================================
        # HANDS
        # ======================================================

        hands = []

        if result.hand_landmarks:

            for landmarks in result.hand_landmarks:

                draw_hand(
                    frame,
                    landmarks,
                    w,
                    h
                )

                hands.append(
                    landmarks
                )


        # ======================================================
        # LOCK / UNLOCK BUTTON
        #
        # Put your INDEX over the tiny button.
        # Then make a THUMB + INDEX PINCH.
        # The toggle fires immediately when the pinch starts.
        # ======================================================

        lock_hovered = False

        if len(hands) >= 1:

            hand = hands[0]

            index_tip = hand[8]
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            lock_hovered = is_inside_lock_button(
                ix,
                iy,
                frame
            )

            ps = palm_size(hand, w, h)

            pinch_distance = distance(
                hand[4],
                hand[8],
                w,
                h
            )

            # More forgiving than the previous threshold.
            pinch_threshold = ps * 0.50

            if (
                lock_hovered
                and
                pinch_distance < pinch_threshold
                and
                not lock_pinch_active
            ):

                now = time.time()

                if now - lock_last_click > 0.60:

                    keyboard_locked = not keyboard_locked

                    # Locked keyboard must remain visible.
                    if keyboard_locked:
                        keyboard_visible = True

                    lock_last_click = now

                lock_pinch_active = True

            # Fingers must separate before another click is armed.
            if pinch_distance > ps * 0.70:
                lock_pinch_active = False

        else:
            lock_pinch_active = False

        # ======================================================
        # OPEN / CLOSE KEYBOARD
        #
        # ONLY WORKS WHEN UNLOCKED
        # ======================================================

        if (
            not keyboard_locked
            and
            len(hands) == 2
        ):

            index1 = hands[0][8]
            index2 = hands[1][8]

            raw_distance = distance(
                index1,
                index2,
                w,
                h
            )

            palm1 = palm_size(
                hands[0],
                w,
                h
            )

            palm2 = palm_size(
                hands[1],
                w,
                h
            )

            average_palm = (
                palm1
                +
                palm2
            ) / 2.0

            normalized_distance = (
                raw_distance
                /
                average_palm
            )


            if smoothed_distance is None:

                smoothed_distance = (
                    normalized_distance
                )

            else:

                smoothed_distance = (
                    smoothed_distance
                    *
                    (
                        1.0
                        -
                        DISTANCE_SMOOTHING
                    )
                    +
                    normalized_distance
                    *
                    DISTANCE_SMOOTHING
                )


            # --------------------------------------------------
            # OPEN
            # --------------------------------------------------

            if (
                not keyboard_visible
                and
                smoothed_distance
                >=
                OPEN_THRESHOLD
            ):

                keyboard_visible = True


            # --------------------------------------------------
            # CLOSE
            # --------------------------------------------------

            elif (
                keyboard_visible
                and
                smoothed_distance
                <=
                CLOSE_THRESHOLD
            ):

                keyboard_visible = False


        else:

            # When locked, distance has ZERO authority.

            if keyboard_locked:

                smoothed_distance = None


        # ======================================================
        # TWO INDEPENDENT INDEX HIGHLIGHTS
        # BOTH hands are required.
        # Each index highlights the key directly underneath it.
        # ======================================================

        hovered_keys = []

        if (
            (keyboard_visible or keyboard_locked)
            and len(hands) >= 2
        ):

            geometry = get_keyboard_geometry(frame)

            for hand in hands[:2]:
                index_tip = hand[8]
                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)

                hovered_key = get_hovered_key(
                    ix,
                    iy,
                    geometry
                )

                if hovered_key is not None:
                    hovered_keys.append(hovered_key)

        else:
            hovered_keys = []

        # ======================================================
        # DRAW KEYBOARD
        # ======================================================

        if keyboard_visible or keyboard_locked:

            draw_keyboard(
                frame,
                hovered_keys
            )


        # ======================================================
        # DRAW LOCK BUTTON
        # ======================================================

        draw_lock_button(
            frame,
            keyboard_locked,
            lock_hovered
        )


        # ======================================================
        # STATUS
        # ======================================================

        if keyboard_locked:

            status = (
                "KEYBOARD LOCKED "
                "| BOTH HANDS AVAILABLE"
            )

        elif keyboard_visible:

            status = (
                "KEYBOARD UNLOCKED "
                "| TWO-FINGER GESTURE ACTIVE"
            )

        else:

            status = (
                "KEYBOARD HIDDEN "
                "| TWO INDEX FINGERS TO OPEN"
            )


        cv2.putText(
            frame,
            status,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 105, 180),
            1,
            cv2.LINE_AA
        )


        # Debug distance

        if (
            DEBUG
            and
            smoothed_distance is not None
            and
            not keyboard_locked
        ):

            cv2.putText(
                frame,
                (
                    f"INDEX DIST: "
                    f"{smoothed_distance:.2f}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 105, 180),
                1,
                cv2.LINE_AA
            )


        # ======================================================
        # SHOW
        # ======================================================

        cv2.imshow(
            "NEXUS - Virtual Keyboard",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            break


    # ==========================================================
    # CLEANUP
    # ==========================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    main()