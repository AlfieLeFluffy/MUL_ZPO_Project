import cv2 # type: ignore

class ImageProcessor:
    def __init__(self):
        pass
    
    def load_image(self, _image_path):
        self.image = cv2.imread(_image_path, cv2.IMREAD_COLOR_RGB)
        return self.image

    def process_image(self):
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def save_image(self, _image, _output_path):
        cv2.imwrite(_output_path, _image)