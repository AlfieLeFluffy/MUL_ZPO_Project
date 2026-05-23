import cv2
import numpy as np
from enum import Enum


class KernelGeneration:
    class KernelType(Enum):
        GAUSS_BLUR = "Gauss Blur"
        BOX_BLUR = "Box Blur"
        SOBEL = "Sobel"

    class FilterType(Enum):
        HIGH_PASS = "High Pass"
        LOW_PASS = "Low Pass"
        HAMMING = "Hamming Window"

    @staticmethod
    def create_kernel(_type, _kernel_size: tuple, _sigma: float):
        """A header method that combines other kernel generation methods

        Args:
            _type (KernelType): Type of kernel
            _kernel_size (tuple): Kernel size
            _sigma (float): Kernel variable

        Returns:
            np.ndarray: Output kernel
        """
        if isinstance(_type, KernelGeneration.KernelType):
            match _type:
                case KernelGeneration.KernelType.GAUSS_BLUR:
                    return KernelGeneration.create_gauss_blur_kernel(
                        _kernel_size, _sigma
                    )
                case KernelGeneration.KernelType.BOX_BLUR:
                    return KernelGeneration.create_box_blur_kernel(_kernel_size, _sigma)
                case KernelGeneration.KernelType.SOBEL:
                    return KernelGeneration.create_sobel_kernel(_kernel_size, _sigma)
        elif isinstance(_type, str):
            match _type:
                case KernelGeneration.KernelType.GAUSS_BLUR.value:
                    return KernelGeneration.create_gauss_blur_kernel(
                        _kernel_size, _sigma
                    )
                case KernelGeneration.KernelType.BOX_BLUR.value:
                    return KernelGeneration.create_box_blur_kernel(_kernel_size, _sigma)
                case KernelGeneration.KernelType.SOBEL.value:
                    return KernelGeneration.create_sobel_kernel(_kernel_size, _sigma)

    @staticmethod
    def create_gauss_blur_kernel(_kernel_size: tuple, _sigma: float):
        assert len(_kernel_size) == 2, (
            "create_gauss_blur_kernel: kernel size has invalid dimensions"
        )
        assert _kernel_size[0] == _kernel_size[1], (
            "create_gauss_blur_kernel: kernel must be square"
        )

        kernel = (
            cv2.getGaussianKernel(_kernel_size[0], _sigma)
            @ cv2.getGaussianKernel(_kernel_size[1], _sigma).T
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def create_box_blur_kernel(_kernel_size: tuple, _sigma: float):
        assert len(_kernel_size) == 2, (
            "create_gauss_blur_kernel: kernel size has invalid dimensions"
        )
        assert _kernel_size[0] == _kernel_size[1], (
            "create_gauss_blur_kernel: kernel must be square"
        )

        kernel = np.ones_like((_kernel_size, _kernel_size))
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def create_sobel_kernel(_kernel_size: tuple, _sigma: float):
        """Create a dynamically sized Sobel kernel

        Args:
            _kernel_size(tuple): Kernel size
            _sigma: Rotation

        Returns:
            np.ndarray: Sobel kernel
        """
        assert len(_kernel_size) == 2, (
            "create_sobel_kernel: kernel size has invalid dimensions"
        )
        assert _kernel_size[0] == _kernel_size[1], (
            "create_sobel_kernel: kernel must be square"
        )

        kernel = np.array(
            [["-1", "0", "1"], ["-2", "0", "2"], ["-1", "0", "1"]], dtype=np.float32
        )

        kernel_size = _kernel_size[0]
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Sobel kernel size must be odd and at least 3")

        if kernel_size > 3:
            kernel = cv2.resize(
                kernel, (kernel_size, kernel_size), interpolation=cv2.INTER_AREA
            )
            kernel[:, 0 : int(kernel_size / 2)] = np.fliplr(
                kernel[:, 0 : int(kernel_size / 2)]
            )
            kernel[:, int(kernel_size / 2) + 1 : kernel_size] = np.fliplr(
                kernel[:, int(kernel_size / 2) + 1 : kernel_size]
            )
            kernel = kernel.astype(np.int8) * int(kernel_size / 2)
        kernel = np.rot90(kernel, int(_sigma))
        kernel = kernel / np.sum(np.abs(kernel))
        return kernel.astype(np.float32)

    @staticmethod
    def create_filter(_type: FilterType, _filter_size: tuple, _offset: int):
        """A header method for aggregating other filter creation methods

        Args:
            _type (FilterType): Filter type
            _filter_size (tuple): Filter size
            _offset (int): Offset within the filter

        Returns:
            np.ndarray: Output filter
        """
        match _type:
            case KernelGeneration.FilterType.HIGH_PASS:
                return KernelGeneration.create_high_pass_filter(_filter_size, _offset)
            case KernelGeneration.FilterType.LOW_PASS:
                return KernelGeneration.create_low_pass_filter(_filter_size, _offset)
            case KernelGeneration.FilterType.HIGH_PASS:
                return KernelGeneration.create_hamming_window(_filter_size, _offset)

    @staticmethod
    def create_high_pass_filter(_filter_size, _offset=50):
        kernel = np.zeros(_filter_size, dtype=np.complex128)
        centerX = _filter_size[0] // 2
        centerY = _filter_size[1] // 2
        if len(_filter_size) == 3:
            for i in range(kernel.shape[2]):
                kernel[:, :, i] = KernelGeneration.create_low_pass_filter(
                    (_filter_size[0], _filter_size[1]), _offset
                )
        else:
            kernel[
                centerX - _offset : centerX + _offset,
                centerY - _offset : centerY + _offset,
            ] = 1
        return kernel

    @staticmethod
    def create_low_pass_filter(_filter_size, _offset=50):
        kernel = np.ones(_filter_size, dtype=np.complex128)
        centerX = _filter_size[0] // 2
        centerY = _filter_size[1] // 2
        if len(_filter_size) == 3:
            for i in range(kernel.shape[2]):
                kernel[:, :, i] = KernelGeneration.create_high_pass_filter(
                    (_filter_size[0], _filter_size[1]), _offset
                )
        else:
            kernel[
                centerX - _offset : centerX + _offset,
                centerY - _offset : centerY + _offset,
            ] = 0
        return kernel

    @staticmethod
    def create_hamming_window(_filter_size, _offset=50):
        kernel = np.zeros(_filter_size, dtype=np.complex128)
        if len(_filter_size) == 3:
            for i in range(kernel.shape[2]):
                kernel[:, :, i] = KernelGeneration.create_hamming_window(
                    (_filter_size[0], _filter_size[1]), _offset
                )
            return kernel

        max_offset_x = min(_offset, _filter_size[0] // 2)
        max_offset_y = min(_offset, _filter_size[1] // 2)
        window_size_x = max_offset_x * 2
        window_size_y = max_offset_y * 2

        if window_size_x <= 0 or window_size_y <= 0:
            return kernel

        hamming_x = np.hamming(window_size_x).astype(np.complex128)
        hamming_y = np.hamming(window_size_y).astype(np.complex128)
        window = np.outer(hamming_x, hamming_y)

        center_x = _filter_size[0] // 2
        center_y = _filter_size[1] // 2
        x_start = center_x - max_offset_x
        y_start = center_y - max_offset_y
        kernel[x_start : x_start + window_size_x, y_start : y_start + window_size_y] = (
            window
        )
        return kernel
