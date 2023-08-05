import cv2
import numpy as np
import os
from helper import *


class MEstimator:
    def __init__(self, 
                 accumulate_flow = True, 
                 glitchy = False, 
                 accumulate_pixels = False, 
                 toroidal = True, 
                 motion_folder = "./motion_folder",
                 input_folder = "./input_folder",
                 output_folder = "./output_folder"):
        self.accumulate_flow = accumulate_flow
        self.accumulate_pixels = accumulate_pixels
        self.toroidal = toroidal
        self.glitchy = glitchy
        self.motion_folder = motion_folder
        self.input_folder = input_folder
        self.output_folder = output_folder
    

    def transfer_motion(self):
        # calculate the flow vectors
        cv2.setRNGSeed(0)
        # default
        flow = cv2.calcOpticalFlowFarneback(self.img1, self.img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        
        # flow = cv2.calcOpticalFlowFarneback(self.img1, self.img2, None, 0.8, 5, 15, 3, 5, 1.2, 0)
        # for fun
        # flow = flow * 5

        # do you want to accumulate the flow throughout the lifetime of the object
        if self.accumulate_flow:
            try:
                self.prev_flow
            except AttributeError:
                self.prev_flow = np.zeros((self.height, self.width, 2), "float32")
            flow = flow + self.prev_flow
            self.prev_flow = flow

        # carry pixels using these flow vectors
        imgTview = self.imgT.view(dtype=[("b", "uint8"), ("g", "uint8"), ("r", "uint8")]).squeeze(2)
        final_indices = self.indices.astype("int32")
        flow_int_round = flow.round().astype("int32")
        final_indices[:,:,0] = final_indices[:,:,0] + flow_int_round[:,:,1]
        final_indices[:,:,1] = final_indices[:,:,1] + flow_int_round[:,:,0]
        final_indices_X = final_indices[:,:, 0]
        final_indices_Y = final_indices[:,:, 1]


        # clip on edges or wrap around
        if self.toroidal:
            final_indices_X %= self.height
            final_indices_Y %= self.width
        else:
            final_indices_X.clip(0, self.height - 1)
            final_indices_Y.clip(0, self.width - 1)

        # glitchy version takes the target pixels to source pixels (assumes the reverse motion)
        if self.glitchy:
            final_view = imgTview[final_indices_X, final_indices_Y]
            final_img = final_view.view(dtype="uint8").reshape(self.height, self.width,3)
        # non-glitchy version takes the source pixels to target pixels (assumes the proper motion)
        else:
            imgTview[final_indices_X, final_indices_Y] = imgTview
            final_img = imgTview.view(dtype="uint8").reshape(self.height, self.width,3)

        # not properly working rn
        # not clear how to do this
        # if self.accumulate_pixels:
        #     try:
        #         self.prev_pixels
        #     except AttributeError:
        #         self.prev_pixels = np.zeros((self.height, self.width, 3), "uint8")
            
        #     index_diff = np.abs(final_indices - self.indices) > 0
        #     index_bool = np.bitwise_or(index_diff[:,:,0], index_diff[:,:,1])
        #     self.prev_pixels[index_bool] = final_img[index_bool]
        #     print(self.prev_pixels[index_bool].shape)
        #     cv2.imshow("", self.prev_pixels)
        #     cv2.waitKey()


        return final_img

    def batch_process(self):
        motion_files = os.listdir(self.motion_folder)
        motion_files = [self.motion_folder + "/" + file for file in motion_files]

        input_files = os.listdir(self.input_folder)
        input_files = [self.input_folder + "/" + file for file in input_files]

        for i in range(min(len(input_files), len(motion_files) - 1)):
        # for i in range(3):
            self.load_images(motion_files[i], motion_files[i+1], input_files[i])
            self.prepare_indices()
            out = self.transfer_motion()
            output_file_path = self.output_folder + "/" + input_files[i][len(self.input_folder):]
            cv2.imwrite(output_file_path, out)

    def load_images(self, inp1_path, inp2_path, trans_path):
        self.img1 = cv2.cvtColor(cv2.imread(inp1_path), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.imread(inp2_path), cv2.COLOR_BGR2GRAY)
        self.imgT = cv2.imread(trans_path)
    
    
    # combine row and column indices into an array of (x, y) coordinates
    def prepare_indices(self):
        self.width = self.img1.shape[1]
        self.height = self.img2.shape[0]
        rows, cols = np.mgrid[range(self.height), range(self.width)]

        self.indices = np.stack((rows, cols), axis = 2)


