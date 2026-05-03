import numpy as np

# from src.util.plotting import plot_images
from src.conv.kernel_generation import KernelGeneration as kg


class SpectralConvolution:
    @staticmethod
    def fft_convolve(_image: np.ndarray, _kernel: np.ndarray, mode: str = "same"):
        """Convolve an image with a kernel using FFT.

        Parameters:
            _image: Input image
            _kernel: Input kernel
            mode: "same" returns output the same size as input image
                  "full" returns the full linear convolution
                  "valid" returns output only where kernel completely overlaps image

        Returns:
            (np.ndarray): Output image
        """
        image = np.asarray(_image, dtype=float)
        kernel = np.asarray(_kernel, dtype=float)

        if image.ndim == 3 and kernel.ndim == 2:
            channels = [
                SpectralConvolution.fft_convolve(image[:, :, c], kernel, mode=mode)
                for c in range(image.shape[2])
            ]
            return np.stack(channels, axis=2)

        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError(
                "fft_convolve supports a 2D image and 2D kernel, or a 3D image with a 2D kernel"
            )

        fft_shape = tuple(np.array(image.shape) + np.array(kernel.shape) - 1)
        image_fft = np.fft.fft2(image, s=fft_shape)
        kernel_fft = np.fft.fft2(kernel, s=fft_shape)
        conv = np.fft.ifft2(image_fft * kernel_fft)
        conv = np.real(conv)

        if mode == "full":
            return conv
        if mode == "same":
            start = [(conv.shape[i] - image.shape[i]) // 2 for i in range(2)]
            end = [start[i] + image.shape[i] for i in range(2)]
            return conv[start[0] : end[0], start[1] : end[1]]
        if mode == "valid":
            start = [kernel.shape[i] - 1 for i in range(2)]
            end = [start[i] + image.shape[i] - kernel.shape[i] + 1 for i in range(2)]
            return conv[start[0] : end[0], start[1] : end[1]]

        raise ValueError("mode must be 'same', 'full', or 'valid'")

    @staticmethod
    def fft_filter(_image: np.ndarray, _type: kg.FilterType, _offset: int):
        """Filters an image with a kernel using FFT.

        Parameters:
            _image: Input image
            _type: Filter type
            _offset: Size of the filter offset

        Returns:
            (np.ndarray): Output image
        """
        image = np.asarray(_image, dtype=float)
        fft_shape = tuple(np.array(image.shape))

        if image.ndim == 3:
            channels = [
                SpectralConvolution.fft_filter(image[:, :, c], _type, _offset)
                for c in range(image.shape[2])
            ]
            return np.stack(channels, axis=2)

        match _type:
            case kg.FilterType.HIGH_PASS:
                kernel = kg.create_high_pass_filter(
                    (fft_shape[0], fft_shape[1]),
                    _offset,
                )
            case kg.FilterType.LOW_PASS:
                kernel = kg.create_low_pass_filter(
                    (fft_shape[0], fft_shape[1]),
                    _offset,
                )
            case kg.FilterType.HAMMING:
                kernel = kg.create_hamming_window(
                    (fft_shape[0], fft_shape[1]),
                    _offset,
                )
            case _:
                raise ValueError("fft_filter: invalid filter type")

        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError(
                "fft_convolve supports a 2D image and 2D kernel, or a 3D image with a 2D kernel"
            )

        image_fft = np.fft.fft2(image, s=fft_shape)
        conv = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(image_fft) * kernel))
        conv = np.real(conv)

        return conv
