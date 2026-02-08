import os
import random

import numpy as np
from numpy import array as ary

"""Numpy for data manipulation"""

import itertools

from sklearn import linear_model as lm

"""Using linear models from Sklearn"""
import pandas as pd

"""Using Pandas data frames"""
import logging as log
from time import sleep, time

# CV
import cv2 as cv
import pygame as pg
from pygame.draw import circle

"""OpenCV computer vision library"""


def main():
    print("main")
    pg.init()
    pg.display.set_mode((800, 800))
    SCREEN = pg.display.get_surface()
    Cal = Calib(SCREEN)
    print(str(Cal.targets))
    print("active: " + str(Cal.active))
    Cal.draw()
    pg.display.update()
    sleep(2)


def draw_text(
    text: str,
    Surf: pg.Surface,
    rel_pos: tuple,
    Font: pg.font.Font,
    color=(0, 0, 0),
    center=False,
):
    surf_size = Surf.get_size()
    x, y = np.array(rel_pos) * np.array(surf_size)
    rendered_text = Font.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and why coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    # This puts the pre-rendered object on the surface
    Surf.blit(rendered_text, box)


class Stimulus:
    stim_dir = "Stimuli/"

    def __init__(self, entry):
        if isinstance(entry, pd.DataFrame):
            entry = entry.to_dict()
        self.file = entry["File"]
        self.path = os.path.join(self.stim_dir, self.file)
        self.size = ary((entry["width"], entry["height"]))

    def load(self, surface: pg.Surface, scale=True):
        image = pg.image.load(self.path)
        # image = pg.image.convert()
        self.surface = surface
        self.surf_size = ary(self.surface.get_size())
        if scale:
            self.scale = min(self.surf_size / self.size)
            scale_to = ary(self.size * self.scale).astype(int)
            self.image = pg.transform.smoothscale(image, scale_to)
            self.size = self.image.get_size()
        else:
            self.scale = 1
        self.pos = ary((self.surf_size - self.size) / 2).astype(int)

    def draw(self):
        self.surface.blit(self.image, self.pos)

    def draw_preview(self):
        blur = ary(self.surf_size / 4).astype("int")  # 10% blur
        img = pg.surfarray.array3d(self.image)
        img = cv.blur(img, blur).astype("uint8")
        # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = pg.surfarray.make_surface(img)
        self.surface.blit(img, self.pos)

    def average_brightness(self):
        return pg.surfarray.array3d(self.image).mean()


class StimulusSet:
    def __init__(self, path):
        self.table = pd.read_csv(path)
        self.Stimuli = []
        for index, row in self.table.iterrows():
            this_stim = Stimulus(row)
            self.Stimuli.append(this_stim)
        self.active = 0

    def n(self):
        return len(self.Stimuli)

    def remaining(self):
        return len(self.Stimuli) - self.active

    def next(self):
        if self.active < len(self.Stimuli):
            this_stim = self.Stimuli[self.active]
            self.active += 1
            return True, this_stim
        else:
            return False, None

    def reset(self):
        self.active = 0

    def pop(self):
        return self.Stimuli.pop()

    def shuffle(self, reset=True):
        self.reset()
        random.shuffle(self.Stimuli)  ## very procedural, brrr


def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


class YETI:
    """
    Class for using a USB camera as an eye tracking device
    ...

    Attributes
    ----------
    usb : int
        number of the USB camera to connect
    device : cv2.VideoCapture
        camera device after connection
    connected : bool
        if camera device is connected
    fps : int
        Frame refresh rate of camera
    frame_size : tuple(int)
        width and height of frames delivered by connected camera
    calib_data : pandas.DataFrame
        Data frame collecting calibration data
    data : pandas.DataFrame
        Data frame to collect recorded eye positions

    surface : pygame.Surface
        a Pygame surface object for drawing
    pro_positions : tuple[int]
        relative target positions used for creating a square calibration surface
    targets : numpy.array
        actual target positions (x,y)
    active : int
        index of the active targets

    Methods
    -------
    release()
        returns the coordinates of the active targets
    init_eye_detection()
        resets the active position to 0
    update_frame()
        returns the number of targets to
    detect_eye()
        returns the number of remaining targets
    update_eye_frame()
        advances active position by 1
    update_quad_bright()
        draws the calibration surface
    record_calib_data()
        adds current measures to calibration data
    train()
        trains the eye tracker using recorded calibration data
    reset()
        resets calibration data, offsets and data
    """

    frame = None
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0, 0, 0, 0)  # make array
    eye_frame = []
    quad_bright = (0, 0, 0, 0)  # make array
    offsets = (0, 0)  # make array
    data_cols = (
        "Exp",
        "Part",
        "Stim",
        "time",
        "xL",
        "yL",
        "xL_pro",
        "yL_pro",
        "xR",
        "yR",
        "xR_pro",
        "yR_pro",
        # keep fused
        "xF",
        "yF",
        "xF_pro",
        "yF_pro",
    )

    def __init__(self, usb: int, surface: pg.Surface) -> None:
        """
        YETI 24 constructor

        usb can be:
            - int (fallback): will use usb and usb+1
            - tuple/list: (usb_left, usb_right)
        """
        self.connected = False
        self.surface = surface
        self.surf_size = self.surface.get_size()

        # Interpreting usb argument
        if isinstance(usb, (tuple, list)) and len(usb) == 2:
            self.usb_L, self.usb_R = int(usb[0]), int(usb[1])
        else:
            self.usb_L = int(usb)
            self.usb_R = int(usb) + 1  # fallback convention

        try:
            self.device_L = cv.VideoCapture(self.usb_L)
            self.device_R = cv.VideoCapture(self.usb_R)
            self.connected = self.device_L.isOpened() and self.device_R.isOpened()
        except:
            log.error(
                f"Could not connect USB devices L={self.usb_L}, R={self.usb_R}: {e}"
            )
            self.connected = False

        # --- init state ---
        self.new_frame = False
        self.frame_L = None
        self.frame_R = None
        self.eye_frame_L = None
        self.eye_frame_R = None

        self.eye_detected_L = False
        self.eye_detected_R = False
        self.eye_frame_coords_L = (0, 0, 0, 0)
        self.eye_frame_coords_R = (0, 0, 0, 0)

        self.frame = None
        self.eye_frame = None

        self.quad_bright = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # this will become 8 values (4 left + 4 right)

        if self.connected:
            self.fps = self.device_L.get(cv.CAP_PROP_FPS)
            self.frame_size = (
                int(self.device_L.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(self.device_L.get(cv.CAP_PROP_FRAME_HEIGHT)),
            )
            self.calib_data = np.zeros(
                shape=(0, 10)
            )  # now 8 features + 2 target coords = 10 columns

            self.data = pd.DataFrame(columns=YETI.data_cols, dtype="float64")
            self.data["Exp"].astype("category")
            self.data["Part"].astype("category")
            self.data["Stim"].astype("category")

            self.update_frame()

    def release(self):  # updated release so close both cameras
        if hasattr(self, "device_L"):
            self.device_L.release()
        if hasattr(self, "device_R"):
            self.device_R.release()

    def init_eye_detection(self, cascade_file: str):
        """
        Initialize eye detection

        :param model Haar cascade file
        """
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

    def update_frame(self) -> np.ndarray:
        """
        Update the eye frame based on eye detection  # now for both cameras
        """
        self.new_frame = False

        new_L, frame_L = self.device_L.read()
        new_R, frame_R = self.device_R.read()

        if (
            new_L
            and new_R
            and (not np.sum(frame_L) == 0)
            and (not np.sum(frame_R) == 0)
        ):
            self.new_frame = True
            self.frame_L = frame_L
            self.frame_R = frame_R

            try:
                h = min(self.frame_L.shape[0], self.frame_R.shape[0])
                L = getattr(self, "debug_L", self.frame_L)
                R = getattr(self, "debug_R", self.frame_R)
                self.frame = np.hstack([L[:h], R[:h]])
            except Exception:
                self.frame = self.frame_L

        return self.new_frame

    def detect_eye(self) -> bool:
        """
        Updates the position and size of the eye frame # now detect both frames
        """
        self.eye_detected_L = False
        self.eye_detected_R = False

        if self.new_frame:
            Eyes_L = self.cascade.detectMultiScale(
                self.frame_L, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )  ## <-- parametrize me
            Eyes_R = self.cascade.detectMultiScale(
                self.frame_R, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )  ## <-- parametrize me
            self.debug_L = self.frame_L.copy()
            self.debug_R = self.frame_R.copy()

            for x, y, w, h in Eyes_L:
                cv.rectangle(self.debug_L, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for x, y, w, h in Eyes_R:
                cv.rectangle(self.debug_R, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(Eyes_L) == 1:
                self.eye_detected_L = True
                self.eye_frame_coords_L = Eyes_L[0]

            if len(Eyes_R) == 1:
                self.eye_detected_R = True
                self.eye_frame_coords_R = Eyes_R[0]

        self.eye_detected = self.eye_detected_L and self.eye_detected_R
        """
        print(
            "L",
            len(Eyes_L),
            "R",
            len(Eyes_R),
            "detL",
            self.eye_detected_L,
            "detR",
            self.eye_detected_R,
        )
        """
        return self.eye_detected

    def update_eye_frame(self) -> np.ndarray:
        """
        Trims the frame obtained by eye detection #now also for both frames
        """

        if not self.new_frame:
            return None

        if self.frame_L is None or self.frame_R is None:
            return None

        if self.new_frame:
            x, y, w, h = self.eye_frame_coords_L
            self.eye_frame_L = self.frame_L[y : y + h, x : x + w]
            self.eye_frame_L = cv.cvtColor(self.eye_frame_L, cv.COLOR_BGR2GRAY)

            x, y, w, h = self.eye_frame_coords_R
            self.eye_frame_R = self.frame_R[y : y + h, x : x + w]
            self.eye_frame_R = cv.cvtColor(self.eye_frame_R, cv.COLOR_BGR2GRAY)

        return self.eye_frame

    def update_quad_bright(self) -> tuple:
        """
        Updating the quadrant brightness vector # now for the 4 quad left + 4 quad right = 8 features -> octant brightness.
        """
        if self.new_frame:

            def quad(img):
                w, h = np.shape(img)
                b_NW = np.mean(img[0 : int(h / 2), 0 : int(w / 2)])
                b_NE = np.mean(img[int(h / 2) : h, 0 : int(w / 2)])
                b_SW = np.mean(img[0 : int(h / 2), int(w / 2) : w])
                b_SE = np.mean(img[int(h / 2) : h, int(w / 2) : w])
                return (b_NW, b_NE, b_SW, b_SE)

            qb_L = quad(self.eye_frame_L)
            qb_R = quad(self.eye_frame_R)

            self.quad_bright = qb_L + qb_R  # length 8

        return self.quad_bright

    def record_calib_data(self, target_pos: tuple) -> ary:
        """
        Record the present quad brightness for training the model

        :param target_pos (x, y) position of calibration target # now record 8 brightness values + (x,y) target = 10 values
        """
        new_data = np.append(self.quad_bright, ary(target_pos))
        self.calib_data = np.append(self.calib_data, [new_data], axis=0)
        return new_data

    """
    def train(self) -> lm.LinearRegression:

        #Trains the eye tracker # now trains using 8 features

        Quad_2 = self.calib_data[:, 0:8]
        Pos = self.calib_data[:, 8:10]
        model = lm.LinearRegression()
        model.fit(Quad_2, Pos)
        self.model = model
        return self.model
    """

    def train(self):
        """
        Train two monocular models:
        - model_L: left 4 features -> (x,y)
        - model_R: right 4 features -> (x,y)
        """
        X_L = self.calib_data[:, 0:4]
        X_R = self.calib_data[:, 4:8]
        Y = self.calib_data[:, 8:10]

        self.model_L = lm.LinearRegression().fit(X_L, Y)
        self.model_R = lm.LinearRegression().fit(X_R, Y)

        return self.model_L, self.model_R

    def update_offsets(self, target_pos: tuple) -> tuple:
        """
        Updates the offset values based on current eye_pos and a given target position

        param target_pos = position of visual target
        """
        # O = E - T
        # T = E - O
        new_offsets = ary(target_pos) - ary(self.eye_raw)  # <---
        self.offsets = tuple(new_offsets)
        return self.offsets

    def reset_offsets(self) -> None:
        self.offsets = (0, 0)

    def update_eye_pos(self) -> tuple:
        """
        Predict per-eye gaze positions (in SCREEN PIXELS, because Calib targets are pixel positions).
        Stores:
          - eye_pos_L, eye_pos_R : (x,y) in pixels
          - eye_pro_L, eye_pro_R : (x,y) in proportions
          - eye_pos (fused)      : optional, for drawing
        """
        quad = ary(self.quad_bright)

        quad_L = quad[0:4].reshape(1, 4)
        quad_R = quad[4:8].reshape(1, 4)

        # raw predictions in screen pixel coordinates
        raw_L = self.model_L.predict(quad_L)[0, :]
        raw_R = self.model_R.predict(quad_R)[0, :]

        self.eye_raw_L = tuple(raw_L)
        self.eye_raw_R = tuple(raw_R)
        self.eye_raw = tuple((ary(self.eye_raw_L) + ary(self.eye_raw_R)) / 2.0)

        # apply offsets (same offsets for both, if you keep quick-cal)
        self.eye_pos_L = tuple(ary(self.eye_raw_L) + ary(self.offsets))
        self.eye_pos_R = tuple(ary(self.eye_raw_R) + ary(self.offsets))

        # proportions relative to screen
        self.eye_pro_L = tuple(ary(self.eye_pos_L) / ary(self.surf_size))
        self.eye_pro_R = tuple(ary(self.eye_pos_R) / ary(self.surf_size))

        # OPTIONAL fused point (not "averaging" unless you call it that)
        # Here: simple mean just for visualization; you can remove if teacher dislikes it.
        self.eye_pos = tuple((ary(self.eye_pos_L) + ary(self.eye_pos_R)) / 2.0)
        self.eye_pro = tuple(ary(self.eye_pos) / ary(self.surf_size))

        return self.eye_pos_L, self.eye_pos_R

    def update_eye_stim(self, Stim: Stimulus) -> tuple:
        """
        Returns the position relative to the stimulus
        """
        offsets = ary(Stim.pos)
        scale = ary(Stim.scale)

        self.eye_stim_L = tuple((ary(self.eye_pos_L) - offsets) / scale)
        self.eye_stim_R = tuple((ary(self.eye_pos_R) - offsets) / scale)

        self.eye_pro_L = tuple(ary(self.eye_stim_L) / ary(Stim.size))
        self.eye_pro_R = tuple(ary(self.eye_stim_R) / ary(Stim.size))

        # fused stim position (optional)
        self.eye_stim = tuple((ary(self.eye_stim_L) + ary(self.eye_stim_R)) / 2.0)
        self.eye_pro = tuple(ary(self.eye_stim) / ary(Stim.size))

        return self.eye_stim_L, self.eye_stim_R

    def record(self, Exp_ID: str, Part_ID: str, Stim_ID: str) -> pd.DataFrame:
        """
        Records the eye coordinates

        """

        new_data = pd.DataFrame(
            {
                "Exp": Exp_ID,
                "Part": Part_ID,
                "Stim": Stim_ID,
                "time": time(),
                "xL": self.eye_stim_L[0],
                "yL": self.eye_stim_L[1],
                "xL_pro": self.eye_pro_L[0],
                "yL_pro": self.eye_pro_L[1],
                "xR": self.eye_stim_R[0],
                "yR": self.eye_stim_R[1],
                "xR_pro": self.eye_pro_R[0],
                "yR_pro": self.eye_pro_R[1],
                # optional fused
                "xF": self.eye_stim[0],
                "yF": self.eye_stim[1],
                "xF_pro": self.eye_pro[0],
                "yF_pro": self.eye_pro[1],
            },
            index=[0],
        )

        self.data = pd.concat([self.data, new_data], ignore_index=True)
        return new_data

    def reset_calib(self) -> None:
        self.calib_data = np.zeros(shape=(0, 10))
        for m in ("model_L", "model_R"):
            if hasattr(self, m):
                delattr(self, m)

    def reset_data(self) -> None:
        self.data = pd.DataFrame(columns=YETI.data_cols, dtype="float64")

    def reset(self) -> None:
        self.reset_calib()
        self.reset_data()

    def draw_follow(self, surface: pg.Surface, add_raw=False, add_stim=False) -> None:
        """
        Draws a circle to the current eye position

        Note that eye positions must be updated using the update methods
        """
        surf_size = ary(surface.get_size())
        circ_size = int(surf_size.min() / 50)
        circ_stroke = int(surf_size.min() / 200)
        circle(surface, (255, 0, 0), self.eye_pos, circ_size, circ_stroke)
        if add_raw:
            circle(surface, (0, 255, 0), self.eye_raw, circ_size, circ_stroke)
        if add_stim:
            circle(surface, (0, 0, 255), self.eye_stim, circ_size, circ_stroke)


class Calib:
    """yeti14 calibration"""

    color = (160, 160, 160)
    active_color = (255, 120, 0)
    radius = 20
    stroke = 10

    """
    Creates a square calib surface using relative positions
    ...

    Attributes
    ----------
    surface : pygame.Surface
        a Pygame surface object for drawing
    pro_positions : tuple[int]
        relative target positions used for creating a square calibration surface
    targets : numpy.array
        actual target positions (x,y)
    active : int
        index of the active targets

    Methods
    -------
    active_pos()
        returns the coordinates of the active targets
    reset()
        resets the active position to 0
    n()
        returns the number of targets to
    remaining()
        returns the number of remaining targets
    next()
        advances active position by 1
    draw()
        draws the calibration surface

    """

    def __init__(self, surface: pg.Surface, pro_positions=(0.125, 0.5, 0.875)) -> None:
        self.surface = surface
        self.surface_size = ary(self.surface.get_size())
        self.pro_positions = ary(pro_positions)
        x_pos = self.pro_positions * self.surface_size[0]
        y_pos = self.pro_positions * self.surface_size[1]
        self.targets = ary(
            list(itertools.product(x_pos, y_pos))
        )  ## No idea how this works
        self.active = 0

    def shuffle(self, reset=True):
        self.reset()
        self.targets = np.random.shuffle(self.targets)

    def active_pos(self) -> int:
        return self.targets[self.active]

    def reset(self) -> None:
        self.active = 0

    def n(self) -> int:
        return len(self.targets[:, 0])

    def remaining(self) -> int:
        return self.n() - self.active - 1

    def next(self) -> tuple:
        if self.remaining():
            this_target = self.targets[self.active]
            self.active += 1
            return True, this_target
        else:
            return False, None

    def draw(self) -> None:
        index = 0
        for target in self.targets:
            pos = list(map(int, target))
            if index == self.active:
                color = self.active_color
            else:
                color = self.color
            index += 1
            circle(self.surface, color, pos, self.radius, self.stroke)
