import cv2
import numpy as np
import os
from me_library import MEstimator
opts = {
    "accumulate_flow": True, 
    "accumulate_pixels": True, # this does nothing right now
    "glitchy": True,
    "motion_folder": "./motion_folder",
    "input_folder": "./input_folder",
    "output_folder": "./output_folder"
}
me = MEstimator(*opts)
me.batch_process()
