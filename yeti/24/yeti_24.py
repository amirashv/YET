"""
YETI24 calibration prototype (2 cams).

Collects quadrant brightness (4 per eye) + target positions from an Excel table,
trains 2 linear regressors (left/right) and shows a fused crosshair for a quick sanity check.

Note: pd.read_excel needs openpyxl.
"""

from time import time

import os
import sys
import logging as log

import cv2 as cv
import numpy as np
import pandas as pd

import pygame as pg
from pygame.locals import QUIT, KEYDOWN, K_SPACE, K_RETURN, K_BACKSPACE

from sklearn import linear_model as lm

# Config
YETI = 24
YETI_NAME = f"Yeti{YETI}"

CONFIG = "yeti/24/yeti_24.xlsx"  # must contain columns: x_L,y_L,x_R,y_R
RESULTS_DIR = "24"
RESULTS = os.path.join(RESULTS_DIR, f"yeti_24_{int(time())}.xlsx")

EYE_CASCADE_FILE = "trained_models/haarcascade_eye.xml"

# Screen
SCREEN_W = 1000
SCREEN_H = 1000
SCREEN_SIZE = (SCREEN_W, SCREEN_H)

# Eye ROI normalization (aligns with newer libyeti24 pipeline)
FIXED_EYE_SIZE = (64, 48)  # (width, height)

# Detection parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (100, 100)  # adjust if needed

# Colors
COL_BLACK = (0, 0, 0)
COL_GRAY = (120, 120, 120)
COL_GREEN = (0, 255, 0)
COL_ACTIVE = (255, 120, 0)


def frame_to_surf(frame_bgr, dim):
    """Convert an OpenCV BGR image to a Pygame Surface (rotated to match pygame coords)."""
    img = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    img = np.rot90(img)
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


def draw_text(screen, font, text, pos, color=(255, 255, 255)):
    rendered = font.render(text, True, color)
    screen.blit(rendered, rendered.get_rect(topleft=pos))


def draw_rect(screen, x, y, w, h, color=(255, 255, 255), stroke=1):
    pg.draw.rect(screen, color, (x, y, w, h), stroke)


def quad_bright(gray_eye):
    """Return (NW, NE, SW, SE) mean brightness for a grayscale eye ROI."""
    h, w = gray_eye.shape
    h2, w2 = h // 2, w // 2
    b_NW = float(np.mean(gray_eye[0:h2, 0:w2]))
    b_NE = float(np.mean(gray_eye[0:h2, w2:w]))
    b_SW = float(np.mean(gray_eye[h2:h, 0:w2]))
    b_SE = float(np.mean(gray_eye[h2:h, w2:w]))
    return (b_NW, b_NE, b_SW, b_SE)


def train_QBG(obs_np):
    """
    Train two linear models from OBS matrix.

    OBS columns:
      [Obs, run,
       NW1, NE1, SW1, SE1,
       NW2, NE2, SW2, SE2,
       x_L, y_L, x_R, y_R]
    """
    quad_L = obs_np[:, 2:6]
    pos_L = obs_np[:, 10:12]
    quad_R = obs_np[:, 6:10]
    pos_R = obs_np[:, 12:14]

    model_L = lm.LinearRegression().fit(quad_L, pos_L)
    model_R = lm.LinearRegression().fit(quad_R, pos_R)
    return model_L, model_R


def drawTargets(screen, targets_df, active=0):
    """Draw calibration target positions as circles (fused midpoint between L/R columns)."""
    radius = 20
    stroke = 10
    for idx, target in targets_df.iterrows():
        color = COL_ACTIVE if idx == active else COL_GRAY
        x = int((target["x_L"] + target["x_R"]) / 2)
        y = int((target["y_L"] + target["y_R"]) / 2)
        pg.draw.circle(screen, color, (x, y), radius, stroke)


def main():
    # Logging
    log.basicConfig(filename="YET.log", level=log.INFO)

    # Check cascade
    if not os.path.isfile(EYE_CASCADE_FILE):
        sys.exit(f"{EYE_CASCADE_FILE} not found. CWD: {os.getcwd()}")
    eyeCascade = cv.CascadeClassifier(EYE_CASCADE_FILE)

    # Read targets (requires openpyxl)
    # pip install openpyxl
    targets = pd.read_excel(CONFIG)
    required_cols = {"x_L", "y_L", "x_R", "y_R"}
    if not required_cols.issubset(set(targets.columns)):
        sys.exit(f"CONFIG missing required columns: {required_cols}. Found: {list(targets.columns)}")

    cam1 = cv.VideoCapture(0)
    cam2 = cv.VideoCapture(1)
    if not (cam1.isOpened() and cam2.isOpened()):
        print("Unable to load one or both cameras.")
        sys.exit(1)

    pg.init()
    pg.display.set_mode(SCREEN_SIZE)
    pg.display.set_caption(YETI_NAME)
    screen = pg.display.get_surface()
    font = pg.font.Font("freesansbold.ttf", 28)

    # State
    STATE = "Detect"  # Detect -> Target -> Measure -> Save -> Train -> Validate
    detected = False

    n_targets = len(targets)
    active_target = 0
    run_id = 0
    obs_id = 0

    # Store last valid eye rectangles (so Validate can keep cropping safely)
    eye1_rect = None  # (x,y,w,h) in cam1 frame
    eye2_rect = None  # (x,y,w,h) in cam2 frame

    # Models
    M_L, M_R = None, None
    last_pos_LR = None  # (pos_L, pos_R)

    # OBS table
    OBS_cols = (
        "Obs", "run",
        "NW1", "NE1", "SW1", "SE1",
        "NW2", "NE2", "SW2", "SE2",
        "x_L", "y_L", "x_R", "y_R",
    )
    OBS = np.empty(shape=(0, len(OBS_cols)), dtype=float)

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or not ret2:
            print("Can't receive frame (stream end?) from one or both cameras. Exiting ...")
            break

        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Detect (only in Detect state)
        if STATE == "Detect":
            eyes1 = eyeCascade.detectMultiScale(
                gray1, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
            )
            eyes2 = eyeCascade.detectMultiScale(
                gray2, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
            )

            # Same policy as libyeti24: require exactly 1 eye in each camera
            if len(eyes1) == 1 and len(eyes2) == 1:
                detected = True
                eye1_rect = tuple(int(v) for v in eyes1[0])
                eye2_rect = tuple(int(v) for v in eyes2[0])
            else:
                detected = False

        # Crop eyes if we have valid rects
        F_eye1 = None
        F_eye2 = None
        if eye1_rect is not None and eye2_rect is not None:
            x1, y1, w1, h1 = eye1_rect
            x2, y2, w2, h2 = eye2_rect

            # Safe bounds
            if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
                roi1 = gray1[y1:y1 + h1, x1:x1 + w1]
                roi2 = gray2[y2:y2 + h2, x2:x2 + w2]

                # Guard against empty ROI (can happen if rect goes out of bounds)
                if roi1.size > 0 and roi2.size > 0:
                    # Normalize ROI size (important for stable brightness features)
                    F_eye1 = cv.resize(roi1, FIXED_EYE_SIZE, interpolation=cv.INTER_AREA)
                    F_eye2 = cv.resize(roi2, FIXED_EYE_SIZE, interpolation=cv.INTER_AREA)

        # Validate prediction
        if STATE == "Validate":
            if (M_L is not None) and (M_R is not None) and (F_eye1 is not None) and (F_eye2 is not None):
                q1 = np.array(quad_bright(F_eye1)).reshape(1, 4)
                q2 = np.array(quad_bright(F_eye2)).reshape(1, 4)

                pos_L = M_L.predict(q1)[0, :]  # [x_L, y_L]
                pos_R = M_R.predict(q2)[0, :]  # [x_R, y_R]
                last_pos_LR = (pos_L, pos_R)
            else:
                last_pos_LR = None

        # Event handling
        for event in pg.event.get():
            if event.type == QUIT:
                cam1.release()
                cam2.release()
                pg.quit()
                sys.exit()

            if event.type == KEYDOWN:
                if STATE == "Detect":
                    if detected and event.key == K_SPACE:
                        STATE = "Target"
                        print(STATE, active_target)

                elif STATE == "Target":
                    if event.key == K_SPACE:
                        # Only allow measuring if we have usable eye ROIs
                        if (F_eye1 is not None) and (F_eye2 is not None):
                            STATE = "Measure"
                            print(STATE, active_target)
                        else:
                            # Force re-detect if we lost ROI
                            STATE = "Detect"
                            print("Lost eye ROI -> back to Detect")

                elif STATE == "Save":
                    if event.key == K_SPACE:
                        STATE = "Train"
                    elif event.key == K_RETURN:
                        os.makedirs(RESULTS_DIR, exist_ok=True)
                        df = pd.DataFrame(OBS, columns=OBS_cols)
                        with pd.ExcelWriter(RESULTS) as writer:
                            df.to_excel(writer, sheet_name=f"Obs_{int(time())}", index=False)
                        print("Saved:", RESULTS)
                    elif event.key == K_BACKSPACE:
                        STATE = "Target"
                        active_target = 0
                        run_id += 1

                elif STATE == "Validate":
                    # Optional: press BACKSPACE to go back and re-run calibration
                    if event.key == K_BACKSPACE:
                        STATE = "Target"
                        active_target = 0
                        run_id += 1

        # Automatic transitions
        if STATE == "Measure":
            # Must have valid ROIs
            if (F_eye1 is None) or (F_eye2 is None):
                STATE = "Detect"
            else:
                obs_id += 1
                this_id = np.array([obs_id, run_id], dtype=float)

                # Targets from table (per-eye)
                this_targ_L = np.array([targets.loc[active_target, "x_L"], targets.loc[active_target, "y_L"]], dtype=float)
                this_targ_R = np.array([targets.loc[active_target, "x_R"], targets.loc[active_target, "y_R"]], dtype=float)

                # Quadrant brightness
                this_bright1 = np.array(quad_bright(F_eye1), dtype=float)
                this_bright2 = np.array(quad_bright(F_eye2), dtype=float)

                # Compose row
                this_obs = np.hstack((this_id, this_bright1, this_bright2, this_targ_L, this_targ_R))
                OBS = np.vstack((OBS, this_obs))

                print("Recorded:", this_obs)

                if (active_target + 1) < n_targets:
                    active_target += 1
                    STATE = "Target"
                else:
                    STATE = "Save"

        if STATE == "Train":
            if OBS.shape[0] >= 3:
                M_L, M_R = train_QBG(OBS)
                STATE = "Validate"
                print("Trained. Entering Validate.")
            else:
                print("Not enough samples to train.")
                STATE = "Target"
                active_target = 0

        # Draw
        screen.fill(COL_BLACK)

        if STATE == "Detect":
            # Show eye ROIs if detected and available, else show full frames
            if detected and (F_eye1 is not None) and (F_eye2 is not None):
                img1 = frame_to_surf(cv.cvtColor(F_eye1, cv.COLOR_GRAY2BGR), (200, 200))
                img2 = frame_to_surf(cv.cvtColor(F_eye2, cv.COLOR_GRAY2BGR), (200, 200))
                draw_text(screen, font, "DETECT: OK (press SPACE)", (30, 30), color=COL_GREEN)
            else:
                img1 = frame_to_surf(frame1, (200, 200))
                img2 = frame_to_surf(frame2, (200, 200))
                draw_text(screen, font, "DETECT: find exactly one eye per camera", (30, 30), color=COL_GRAY)

            screen.blit(img1, (200, 400))
            screen.blit(img2, (600, 400))

        elif STATE == "Target":
            if detected:
                draw_text(screen, font, f"TARGET {active_target+1}/{n_targets} (press SPACE)", (30, 30), color=COL_GREEN)
                drawTargets(screen, targets, active_target)
            else:
                draw_text(screen, font, "Lost detection -> returning to Detect", (30, 30), color=COL_GRAY)
                STATE = "Detect"

        elif STATE == "Save":
            draw_text(screen, font, "SAVE: ENTER to write Excel, SPACE to train, BACKSPACE to restart", (30, 30), color=COL_GRAY)
            draw_text(screen, font, f"Samples: {OBS.shape[0]}", (30, 70), color=COL_GRAY)

        elif STATE == "Validate":
            draw_text(screen, font, "VALIDATE: live prediction (BACKSPACE = restart)", (30, 30), color=COL_GREEN)

            if last_pos_LR is not None:
                pos_L, pos_R = last_pos_LR
                fused = (pos_L + pos_R) / 2.0

                x = int(np.clip(fused[0], 0, SCREEN_W - 1))
                y = int(np.clip(fused[1], 0, SCREEN_H - 1))

                # Crosshair
                draw_rect(screen, x, 0, 2, SCREEN_H, color=COL_GREEN, stroke=1)
                draw_rect(screen, 0, y, SCREEN_W, 2, color=COL_GREEN, stroke=1)

                # Diagnostics (actual scalar values)
                draw_text(screen, font, f"Left:  ({pos_L[0]:.1f}, {pos_L[1]:.1f})", (30, 70), color=COL_GRAY)
                draw_text(screen, font, f"Right: ({pos_R[0]:.1f}, {pos_R[1]:.1f})", (30, 110), color=COL_GRAY)
                draw_text(screen, font, f"Fused: ({x}, {y})", (30, 150), color=COL_GREEN)
            else:
                draw_text(screen, font, "No valid ROI/model -> go back to Detect/Train", (30, 70), color=COL_GRAY)

        pg.display.update()

    # Cleanup
    cam1.release()
    cam2.release()
    pg.quit()


if __name__ == "__main__":
    main()
