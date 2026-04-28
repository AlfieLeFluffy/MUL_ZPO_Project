import cv2
import numpy as np

class KernelGeneration:
    @staticmethod
    def create_low_pass_filter(_kernel_size, _offset=50):
        kernel = np.zeros(_kernel_size, dtype=np.float32)
        centerX = _kernel_size[0] // 2
        centerY = _kernel_size[1] // 2
        if len(_kernel_size) == 3:
            for i in range(kernel.shape[2]):
                kernel[:,:,i] = KernelGeneration.create_low_pass_filter((_kernel_size[0], _kernel_size[1]), _offset)
        else:
            kernel[centerX-_offset:centerX+_offset, centerY-_offset:centerY+_offset] = 1
            kernel = kernel / np.sum(kernel)
        return kernel
    
    @staticmethod
    def create_high_pass_filter(_kernel_size, _offset=50):
        kernel = np.ones(_kernel_size, dtype=np.float32)
        centerX = _kernel_size[0] // 2
        centerY = _kernel_size[1] // 2
        if len(_kernel_size) == 3:
            for i in range(kernel.shape[2]):
                kernel[:,:,i] = KernelGeneration.create_high_pass_filter((_kernel_size[0], _kernel_size[1]), _offset)
        else:
            kernel[centerX-_offset:centerX+_offset, centerY-_offset:centerY+_offset] = 0
        return kernel
    
    @staticmethod
    def create_hamming_window(_kernel_size):
        if len(_kernel_size) == 3:
            kernel = np.zeros(_kernel_size, dtype=np.float32)
            for i in range(kernel.shape[2]):
                kernel[:,:,i] = KernelGeneration.create_hamming_window((_kernel_size[0], _kernel_size[1]))
            return kernel
        else:
            return np.sqrt(np.outer(np.hamming(_kernel_size[0]), np.hamming(_kernel_size[1])))

    @staticmethod
    def convert_kernel_to_mask(_kernel, _image_size):
        rows, columns = _image_size
        mask = np.dstack((_kernel, _kernel))
        mask = cv2.dft(mask, flags=cv2.DFT_COMPLEX_OUTPUT, dst=_kernel)
        mask = np.fft.fftshift(mask)
        mask = cv2.resize(mask, (columns, rows))
        return mask
