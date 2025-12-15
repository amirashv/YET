## YETI 24: Collecting 2-dim calibration data for quadrant SBG of two cameras
## input = calibration point table
## Results = table with target coordinates and quadrant brightness

from time import time

YETI = 24
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Multi-point calibration measures with octant brightness"
AUTHOR = "A SHYMBOLATOVA"
CONFIG = "yeti/24/yeti_24.xlsx"
RESULTS = "24/yeti_24_" + str(time()) + ".xlsx"

import csv

# DS
import datetime as dt
import logging as log
import os
import random
import sys

# CV
import cv2 as cv
import numpy as np
import pandas as pd

# PG
import pygame as pg

# from pygame.compat import unichr_, unicode_
from pygame.locals import *
from sklearn import linear_model as lm

##### Preparations #####

# CV
log.basicConfig(filename="YET.log", level=log.INFO)
YET1 = cv.VideoCapture(0)  # Open first camera stream
YET2 = cv.VideoCapture(1)  # Open second camera stream
if YET1.isOpened() and YET2.isOpened():
    width1 = int(YET1.get(cv.CAP_PROP_FRAME_WIDTH))
    height1 = int(YET1.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps1 = YET1.get(cv.CAP_PROP_FPS)
    dim1 = (width1, height1)
    width2 = int(YET2.get(cv.CAP_PROP_FRAME_WIDTH))
    height2 = int(YET2.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps2 = YET2.get(cv.CAP_PROP_FPS)
    dim2 = (width2, height2)
else:
    print("Unable to load one or both cameras.")
    exit()

# Reading the CV model for eye detection
eyeCascadeFile = "trained_models/haarcascade_eye.xml"
if os.path.isfile(eyeCascadeFile):
    eyeCascade = cv.CascadeClassifier(eyeCascadeFile)
else:
    sys.exit(eyeCascadeFile + " not found. CWD: " + os.getcwd())

# Reading calibration point matrix from Excel table
Targets = pd.read_excel(CONFIG)

col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_red = (250, 0, 0)
col_green = (0, 255, 0)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)

col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120, 120, 0)

## width and height in pixel
SCREEN_W = 1000
SCREEN_H = 1000
SCREEN_SIZE = (SCREEN_W, SCREEN_H)


pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETI_NAME)
FONT = pg.font.Font("freesansbold.ttf", 40)

SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 60)


def main():
    ## Initial State
    STATE = "Detect"  # Measure, Target
    DETECTED = False
    BACKGR_COL = col_black

    n_targets = len(Targets)
    active_target = 0
    run = 0
    H_offset, V_offset = (0, 0)
    this_pos = (0, 0)

    Eyes = []
    OBS_cols = (
        "Obs",
        "run",
        "NW1",
        "NE1",
        "SW1",
        "SE1",  # Camera 1 quadrants
        "NW2",
        "NE2",
        "SW2",
        "SE2",  # Camera 2 quadrants
        "x_L",
        "y_L",  # Left eye screen coords
        "x_R",
        "y_R",  # Right eye screen coords
    )
    OBS = np.empty(shape=(0, len(OBS_cols)))
    obs = 0

    ## FAST LOOP
    while True:
        # General frame processing
        ret1, Frame1 = (
            YET1.read()
        )  # Frames are converted to grayscale individually for eye detection
        ret2, Frame2 = YET2.read()
        if not ret1 or not ret2:
            print(
                "Can't receive frame (stream end?) from one or both cameras. Exiting ..."
            )
            break
        F_gray1 = cv.cvtColor(Frame1, cv.COLOR_BGR2GRAY)
        F_gray2 = cv.cvtColor(Frame2, cv.COLOR_BGR2GRAY)

        if (
            STATE == "Detect"
        ):  # Detect eyes separately in each frame and crop the eye regions into F_eye1 and F_eye2
            Eyes1 = eyeCascade.detectMultiScale(
                F_gray1, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            Eyes2 = eyeCascade.detectMultiScale(
                F_gray2, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            if len(Eyes1) > 0 and len(Eyes2) > 0:
                DETECTED = True
                x_eye1, y_eye1, w_eye1, h_eye1 = Eyes1[0]
                F_eye1 = F_gray1[y_eye1 : y_eye1 + h_eye1, x_eye1 : x_eye1 + w_eye1]
                x_eye2, y_eye2, w_eye2, h_eye2 = Eyes2[0]
                F_eye2 = F_gray2[y_eye2 : y_eye2 + h_eye2, x_eye2 : x_eye2 + w_eye2]
            else:
                DETECTED = False
                # F_eye = F_gray
                # w_eye, h_eye = (width, height)
        else:
            F_eye1 = F_gray1[y_eye1 : y_eye1 + h_eye1, x_eye1 : x_eye1 + w_eye1]
            F_eye2 = F_gray2[y_eye2 : y_eye2 + h_eye2, x_eye2 : x_eye2 + w_eye2]

        if STATE == "Validate":
            quad1 = np.array(quad_bright(F_eye1))
            quad1.shape = (1, 4)

            quad2 = np.array(quad_bright(F_eye2))
            quad2.shape = (1, 4)

            pos_L = M_L.predict(quad1)[0, :]  # x_L, y_L
            pos_R = M_R.predict(quad2)[0, :]  # x_R, y_R

            this_pos = (pos_L, pos_R)
            print("Left eye:", pos_L, "Right eye:", pos_R)

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Detect":
                if DETECTED:
                    if (
                        event.type == KEYDOWN and event.key == K_SPACE
                    ):  # deleted cause i already detected and cropped eyes above (F_eye1 and F_eye2)
                        STATE = "Target"
                        print(STATE + str(active_target))

            elif STATE == "Target":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure"
                    print(STATE + str(active_target))
            elif STATE == "Save":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Train"
                if event.type == KEYDOWN and event.key == K_RETURN:
                    STATE = "Save"
                    OBS = pd.DataFrame(OBS, columns=OBS_cols)
                    with pd.ExcelWriter(RESULTS) as writer:
                        print(OBS)
                        OBS.to_excel(
                            writer, sheet_name="Obs_" + str(time()), index=False
                        )
                        print(RESULTS)
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target"
                    active_target = 0  # reset
                    run = run + 1
        if event.type == QUIT:
            YET1.release()
            YET2.release()
            pg.quit()
            sys.exit()

        # Automatic transitionals
        if STATE == "Measure":
            obs = obs + 1
            this_id = np.array([obs, run])
            # Get per-eye target coordinates from the Targets table
            # Make sure your Targets dataframe has columns: x_L, y_L, x_R, y_R
            this_targ_L = np.array(
                [Targets.loc[active_target, "x_L"], Targets.loc[active_target, "y_L"]]
            )
            this_targ_R = np.array(
                [Targets.loc[active_target, "x_R"], Targets.loc[active_target, "y_R"]]
            )
            # Quadrant brightness per eye
            this_bright1 = quad_bright(F_eye1)  # left eye
            this_bright2 = quad_bright(F_eye2)  # right eye
            # Combine id, brightness, and targets into one row
            this_obs = np.hstack(
                (this_id, this_bright1, this_bright2, this_targ_L, this_targ_R)
            )

            print(this_obs)
            OBS = np.vstack((OBS, this_obs))

            if (active_target + 1) < n_targets:
                active_target = active_target + 1
                STATE = "Target"
                print(STATE + str(active_target))
            else:
                print(OBS)
                STATE = "Save"

        if STATE == "Train":
            M_L, M_R = train_QBG(OBS)
            STATE = "Validate"

        # Presentitionals
        # over-paint previous with background xcolor
        pg.display.get_surface().fill(BACKGR_COL)

        if STATE == "Detect":
            if DETECTED:
                Img1 = frame_to_surf(F_eye1, (200, 200))
                Img2 = frame_to_surf(F_eye2, (200, 200))
            else:
                Img1 = frame_to_surf(Frame1, (200, 200))
                Img2 = frame_to_surf(Frame2, (200, 200))
            SCREEN.blit(Img1, (200, 400))
            SCREEN.blit(Img2, (600, 400))
        elif STATE == "Target":
            if DETECTED:
                drawTargets(SCREEN, Targets, active_target)
        elif STATE == "Validate":
            msg = "Press Space one time to stop validating, two times for saving. Backspace for back."

            this_pos = np.asarray(this_pos)

            # If this_pos is [[Lx, Ly], [Rx, Ry]] then fuse it:
            if this_pos.shape == (2, 2):
                left = this_pos[0]
                right = this_pos[1]
                fused = (left + right) / 2.0
            else:
                # fallback: assume it's already [x, y]
                fused = this_pos

            x = int(np.clip(fused[0] + H_offset, 0, SCREEN_W - 1))
            y = int(np.clip(fused[1] + V_offset, 0, SCREEN_H - 1))

            draw_rect(
                x,
                0,
                2,
                SCREEN_H,
                stroke_size=1,
                color=col_green,
            )

            draw_rect(
                0,
                y,
                SCREEN_W,
                2,
                stroke_size=1,
                color=col_green,
            )
            # diagnostics
            draw_text(
                "HPOS: " + str(np.round(this_pos[0])), (510, 250), color=col_green
            )
            draw_text(
                "VPOS: " + str(np.round(this_pos[1])), (510, 300), color=col_green
            )
            draw_text("XOFF: " + str(H_offset), (510, 350), color=col_green)
            draw_text("YOFF: " + str(V_offset), (510, 400), color=col_green)

        # update the screen to display the changes you made
        pg.display.update()


def drawTargets(screen, Targets, active=0):
    for index, target in Targets.iterrows():
        color = (160, 160, 160)
        radius = 20
        stroke = 10
        if index == active:
            color = (255, 120, 0)
        x = int((target["x_L"] + target["x_R"]) / 2)
        y = int((target["y_L"] + target["y_R"]) / 2)
        pos = (x, y)
        pg.draw.circle(screen, color, pos, radius, stroke)


## Converts a cv framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


def draw_text(text, dim, color=(255, 255, 255), center=False):
    x, y = dim
    rendered_text = FONT.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and why coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    # This puts a pre-rendered object to the screen
    SCREEN.blit(rendered_text, box)


def draw_rect(x, y, width, height, color=(255, 255, 255), stroke_size=1):
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)


def quad_bright(frame):
    h, w = np.shape(frame)
    b_NW = np.mean(frame[0 : int(h / 2), 0 : int(w / 2)])
    b_NE = np.mean(frame[int(h / 2) : h, 0 : int(w / 2)])
    b_SW = np.mean(frame[0 : int(h / 2), int(w / 2) : w])
    b_SE = np.mean(frame[int(h / 2) : h, int(w / 2) : w])
    out = (b_NW, b_NE, b_SW, b_SE)
    # out = np.array((b_NW, b_NE, b_SW, b_SE))
    # out.shape = (1,4)
    return out


def train_QBG(Obs):
    # Left eye
    Quad_L = Obs[:, 2:6]  # NW1, NE1, SW1, SE1
    Pos_L = Obs[:, 10:12]  # x_L, y_L
    model_L = lm.LinearRegression()
    model_L.fit(Quad_L, Pos_L)
    # Right eye
    Quad_R = Obs[:, 6:10]  # NW2, NE2, SW2, SE2
    Pos_R = Obs[:, 12:14]  # x_R, y_R
    model_R = lm.LinearRegression()
    model_R.fit(Quad_R, Pos_R)
    return model_L, model_R


# Predicts position based on quad-split
def predict_pos(data, model):
    predictions = model.predict(data)
    return predictions


main()
