import cv2  # type: ignore
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

    def __init__(
        self,
        _data: np.array,
        _name: str = None,
        _type: IMAGE_TYPE = IMAGE_TYPE.RGB_INT,
        _stamp: bool = False,
    ):
        self.data = _data
        if _stamp:
            sections = _name.split("_")
            if len(sections) <= 1:
                self.name = "1_" + _name
            else:
                if sections[0].isdigit():
                    self.name = str(int(sections[0]) + 1) + "_" + "_".join(sections[1:])
                else:
                    self.name = "1_" + _name
        else:
            self.name = _name
        self.type = _type
        self.size = _data.shape
        self.saved = False

    def __str__(self):
        return f"CImage(name={self.name}, type={self.type}, size={self.size}, saved={self.saved})"

    def is_saved(self):
        return self.saved

    def save_image(self, _filepath):
        img_bgr = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        check = cv2.imwrite(_filepath, img_bgr)
        assert check is True, f"Failed to save image to path {_filepath}"
        self.saved = True

    @staticmethod
    def load_image(_filepath: str):
        img_bgr = cv2.imread(_filepath, cv2.IMREAD_COLOR_BGR)
        assert img_bgr is not None, f"Failed to load image from path: {_filepath}"

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_name = _filepath.split("/")[-1]
        img = CImage(img_rgb, img_name, CImage.IMAGE_TYPE.RGB_INT)
        img.saved = True
        return img

    def to_double(self):
        match self.type:
            case self.IMAGE_TYPE.RGB_INT:
                data = self.data / 255
                type = self.IMAGE_TYPE.RGB_DOUBLE
                return CImage(data, self.name, type)

    def ycbcr2rgb(self):
        assert self.type in [self.IMAGE_TYPE.YCBCR_INT, self.IMAGE_TYPE.YCBCR_DOUBLE], (
            f"Failed to convert rgb to ycbcr as the image {self.name} is in type {self.type}"
        )

        image_ycbcr = self.data.copy().astype(np.float32)
        image_ycrcb = image_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
        image_rgb = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCR_CB2RGB)
        image_rgb = np.round(image_rgb * 255)
        image_rgb = np.clip(image_rgb, a_min=0, a_max=255)
        image_rgb = image_rgb.astype(np.uint8)

        output = CImage(image_rgb, self.name, self.IMAGE_TYPE.RGB_INT)
        return output

    def rgb2ycbcr(self):
        assert self.type in [self.IMAGE_TYPE.RGB_INT, self.IMAGE_TYPE.RGB_DOUBLE], (
            f"Failed to convert rgb to ycbcr as the image {self.name} is in type {self.type}"
        )

        image_rgb = self.data.copy().astype(np.float32)
        image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCR_CB)
        image_ycbcr = image_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
        image_ycbcr[:, :, 0] = (
            image_ycbcr[:, :, 0] * (235 - 16) + 16
        ) / 255.0  # to [16/255, 235/255]
        image_ycbcr[:, :, 1:] = (
            image_ycbcr[:, :, 1:] * (240 - 16) + 16
        ) / 255.0  # to [16/255, 240/255]

        output = CImage(image_ycbcr, self.name, self.IMAGE_TYPE.YCBCR_DOUBLE)
        return output
