import cv2 # type: ignore
import os 
import tkinter as tk
import numpy as np
from enum import Enum

class CImage:

    class IMAGE_TYPE(Enum):
        RGB_INT = 1
        RGB_DOUBLE = 2
        BGR_INT = 3
        BGR_DOUBLE = 4
        YCBCR_INT = 5
        YCBCR_DOUBLE = 6

    def __init__(self, _data: np.array, _name: str = None, _type: IMAGE_TYPE = IMAGE_TYPE.RGB_INT):
        self.data = _data
        self.name = _name
        self.type = _type
        self.size = _data.shape
    
    def save_image(self, _filepath):
        img_bgr = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        check = cv2.imwrite(_filepath, img_bgr)
        assert check is True, f"Failed to save image to path {_filepath}"

    @staticmethod
    def load_image(_filepath: str):
        img_bgr= cv2.imread(_filepath, cv2.IMREAD_COLOR_BGR)
        assert img_bgr is not None, f"Failed to load image from path: {_filepath}"
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_name = _filepath.split("/")[-1]
        img = CImage(img_rgb, img_name, CImage.IMAGE_TYPE.RGB_INT)
        return img
    
    def to_double(self):
        match self.type:
            case self.IMAGE_TYPE.RGB_INT:
                self.data = self.data / 255
                self.type = self.IMAGE_TYPE.RGB_DOUBLE

    def rgb2ycbcr(self):
        
        im_rgb = self.data.astype(np.float32)
        im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
        im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
        im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
        im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
        self.data = im_ycbcr
        self.type = self.IMAGE_TYPE.YCBCR_DOUBLE
        return self