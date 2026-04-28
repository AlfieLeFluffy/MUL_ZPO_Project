import cv2
import numpy as np
from scipy.signal import fftconvolve

from src.util.math import norm1, norm2
from src.util.plotting import plot_images


class MinRankKernel:
    """An implementation of the Min-Rank-Kernel algorithm contained within a class. All functions are static and the algorithm does not require an instance."""

    class MinRankKernelParam:
        """Parameter class for the Min-Rank-Kernel algorithms."""

        def __init__(self):
            self.tx = 1e-2
            self.tp = 1e-4
            self.tau = 1e-5  # 3e-4 #1e-5
            self.delta = 1e-5

            # total x-k iters
            self.iter_max = 5

            # x step outer iter
            self.opt_x_iter_max = 2

            # x step inner iter
            self.opt_x_jter_max = 2

            # k step kernel  iter
            self.opt_k_iter_max = 3
            self.ep = 1e-3

            # k step  low rank step
            self.opt_r_iter_max = 3

            # flag whether using low rank
            self.low_rank_optimalization = True

            # image reg param
            self.lamb = 80

            # kernel threshold (threshold * max)
            self.threshold = 0.05
            self.kernel_tresholding = 1

            # k step total iter
            self.mu = 1
            self.opt_k_ter_rank = 10

            # whether show inter results (suggested true to tune...)
            self.verbose = True

    @staticmethod
    def deconvolve(
        _image: np.ndarray,
        _ksize: int,
        params: MinRankKernelParam,
    ):
        """The main function for the Min-Rank-Kernel algorithm. Takes in an image, expected kernel size and runtime parameters.

        Args:
            _image (np.ndarray): Input image
            _ksize (int): Expected kernel size
            params (MinRankKernelParam): Algorithm parameters

        Returns:
            np.ndarray: Expected kernel
        """
        assert _image is not None, "ERROR in Min-Rank-Kernel: Missing image!"
        assert np.mod(_ksize, 2) == 1, (
            "ERROR in Min-Rank-Kernel: Kernel size must be odd!"
        )

        kernel_size = _ksize
        minimum_kernel_scale = int(
            (np.max((2 * np.floor((kernel_size - 1) / 32) + 1, 3)))
        )
        kernel_layer = minimum_kernel_scale
        kernel_layer_step = np.sqrt(2)
        kernel_scales = []
        i = 1
        while kernel_layer < kernel_size:
            kernel_scales.append(kernel_layer)
            i += 1
            kernel_layer = int(np.floor(kernel_layer * kernel_layer_step))
            if np.mod(kernel_layer, 2) == 0:
                kernel_layer += 1
        kernel_scales.append(kernel_size)
        assert len(kernel_scales) > 1, (
            "ERROR in Min-Rank-Kernel: Cannot create kernel scales!"
        )

        current_kernel = np.zeros((minimum_kernel_scale, minimum_kernel_scale))
        kernel_centre = int(minimum_kernel_scale / 2)
        current_kernel[kernel_centre, kernel_centre] = 0.5
        current_kernel[kernel_centre, kernel_centre - 1] = 0.5

        x1 = np.zeros((minimum_kernel_scale, minimum_kernel_scale))
        x2 = np.zeros((minimum_kernel_scale, minimum_kernel_scale))

        for i in range(len(kernel_scales)):
            current_layer_kernel_size = kernel_scales[i]
            print(f"Processing ksize = {current_layer_kernel_size}")
            ratio = current_layer_kernel_size / kernel_size
            resize_size = (
                int(np.floor(_image.shape[1] * ratio)),
                int(np.floor(_image.shape[0] * ratio)),
            )
            resize_size_x = (resize_size[0] - 1, resize_size[1] - 1)
            resized_y = cv2.resize(
                _image, resize_size, interpolation=cv2.INTER_AREA
            ).astype(np.double)
            x1 = cv2.resize(x1, resize_size_x, interpolation=cv2.INTER_AREA).astype(
                np.double
            )
            x2 = cv2.resize(x2, resize_size_x, interpolation=cv2.INTER_AREA).astype(
                np.double
            )

            if i != 0:
                current_kernel = cv2.resize(
                    current_kernel,
                    (int(current_layer_kernel_size), int(current_layer_kernel_size)),
                    interpolation=cv2.INTER_LINEAR,
                )

            dx = np.array([[1, -1], [0, 0]])
            dy = np.array([[1, 0], [-1, 0]])

            l2_norm = 6 * current_layer_kernel_size / kernel_scales[0]

            y1 = fftconvolve(resized_y, dx, "valid")
            y2 = fftconvolve(resized_y, dy, "valid")

            x1, x2, current_kernel = MinRankKernel.MRK_deconvolution_cycle(
                y1,
                y2,
                x1,
                x2,
                current_kernel,
                (params.tau * ((i + 1) / len(kernel_scales))),
                l2_norm,
                params,
            )
            y1, x1, current_kernel = MinRankKernel.MRK_recenter_kernel(
                y1, x1, current_kernel, params
            )
            y2, x2, current_kernel = MinRankKernel.MRK_recenter_kernel(
                y2, x2, current_kernel, params
            )

            if params.verbose:
                plot_images(
                    {"x1": x1, "x2": x2, "kernel": current_kernel, "y1": y1, "y2": y2},
                    3,
                    2,
                )

        current_kernel = MinRankKernel.normalize_kernel(current_kernel, params)

        print("--- Min-Rank-Kernel Finished ---")
        return current_kernel

    @staticmethod
    def MRK_recenter_kernel(
        _y: np.ndarray,
        _x: np.ndarray,
        _kernel: np.ndarray,
        params: MinRankKernelParam,
    ):
        """Recenters kernel based on changes in images.

        Args:
            _y (np.ndarray): Output image
            _x (np.ndarray): Input image
            _kernel (np.ndarray): Kernel
            params (MinRankKernelParam): Algorithm parameters

        Returns:
            np.ndarray: Recentered kernel
        """
        kernel_shape = _kernel.shape
        mu_y = np.sum((range(1, kernel_shape[0] + 1) * np.sum(_kernel, 1).T))
        mu_x = np.sum((range(1, kernel_shape[1] + 1) * np.sum(_kernel, 0)))

        offset_x = int(np.round(np.floor(kernel_shape[1] / 2) + 1 - mu_x))
        offset_y = int(np.round(np.floor(kernel_shape[0] / 2) + 1 - mu_y))

        if params.verbose:
            print(
                f"CenterKernel: weightedMean[{mu_x - 1},{mu_y - 1}] offset[{offset_x},{offset_y}]"
            )

        shift_kernel = np.zeros((np.abs(offset_y * 2) + 1, np.abs(offset_x * 2) + 1))
        shift_kernel[np.abs(offset_y) + offset_y, np.abs(offset_x) + offset_x] = 1

        kshift = fftconvolve(_kernel, shift_kernel, "same")
        xshift = fftconvolve(_x, np.flip(np.flip(shift_kernel, 0), 1), "same")
        yshift = fftconvolve(_y, np.flip(np.flip(shift_kernel, 0), 1), "same")

        return yshift, xshift, kshift

    @staticmethod
    def MRK_deconvolution_cycle(
        y1: np.ndarray,
        y2: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        kernel: np.ndarray,
        tau: float,
        L2norm: float,
        params: MinRankKernelParam,
    ):
        """One cycle of the Min-Rank-Kernel algorithm.

        Args:
            y1 (np.ndarray): Resized output image 1
            y2 (np.ndarray): Resized output image 2
            x1 (np.ndarray): Resized convoluted input image 1
            x2 (np.ndarray): Resized convoluted input image 2
            kernel (np.ndarray): Current kernel
            tau (float): Optimalization parameter
            L2norm (float): Optimalization parameter
            params (MinRankKernelParam): Algorithm parameters

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): Input image 1, Input image 2, Kernel
        """
        if np.sum(np.abs(x1)) == 0:
            x1 = y1
        if np.sum(np.abs(x2)) == 0:
            x2 = y2

        x1 = x1 * L2norm / norm2(x1)
        x2 = x2 * L2norm / norm2(x2)

        kernel_size = kernel.shape[0]
        boundary_size = int(np.floor(kernel_size / 2))
        y1v = y1[
            boundary_size : y1.shape[0] - boundary_size,
            boundary_size : y1.shape[1] - boundary_size,
        ]
        y2v = y2[
            boundary_size : y2.shape[0] - boundary_size,
            boundary_size : y2.shape[1] - boundary_size,
        ]

        for i in range(params.iter_max):
            if params.verbose:
                print(f"Iteration {i + 1}...", flush=True)
            x1, x2 = MinRankKernel.MRK_optimize_x(x1, x2, kernel, y1, y2, params)
            for iter in range(params.opt_k_ter_rank):
                if iter == 0:
                    tmpmu = 0
                else:
                    tmpmu = (
                        params.kernel_tresholding
                        * np.exp(iter)
                        / np.exp(params.opt_k_ter_rank)
                    )

                kernel = MinRankKernel.MRK_optimize_kernel(
                    x1, x2, kernel, y1v, y2v, tmpmu, params
                )

                if params.low_rank_optimalization:
                    kernel = MinRankKernel.MRK_optimize_rank(kernel, tau, params)

                kernel[kernel < 0] = 0
                kernel = kernel / np.sum(kernel)

            if params.threshold:
                kernel[
                    kernel
                    < (
                        np.max(kernel)
                        * params.threshold
                        * ((i + 1) / (params.iter_max + 1))
                    )
                ] = 0
            else:
                kernel[kernel < 0] = 0
            kernel = kernel / np.sum(kernel)

        return x1, x2, kernel

    @staticmethod
    def MRK_optimize_x(
        _x1: np.ndarray,
        _x2: np.ndarray,
        _k: np.ndarray,
        _y1: np.ndarray,
        _y2: np.ndarray,
        params: MinRankKernelParam,
    ):
        """Optimalization of x (input image).

        Args:
            _x1 (np.ndarray): Input image 1
            _x2 (np.ndarray): Input image 2
            _k (np.ndarray): Kernel
            _y1 (np.ndarray): Output image 1
            _y2 (np.ndarray): Output image 2
            params (MinRankKernelParam): Algorithm parameters

        Returns:
            (np.ndarray, np.ndarray): Input image 1, Input image 2
        """
        x1 = _x1 * 6 / norm2(_x1)
        x2 = _x2 * 6 / norm2(_x2)

        cost0 = MinRankKernel.get_cost(x1, x2, _y1, _y2, _k, params)

        tx = params.tx
        tp = params.tp

        while tx > tp:
            x1_backup = x1.copy()
            x2_backup = x2.copy()
            for _ in range(params.opt_x_iter_max):
                lx1 = norm2(x1)
                lx2 = norm2(x2)
                for _ in range(params.opt_x_jter_max):
                    gradient1 = params.lamb * fftconvolve(
                        fftconvolve(x1, _k, "same") - _y1, np.rot90(_k, 2), "same"
                    )
                    gradient2 = params.lamb * fftconvolve(
                        fftconvolve(x2, _k, "same") - _y2, np.rot90(_k, 2), "same"
                    )

                    tmp1 = x1 - tx * lx1 * gradient1
                    tmp2 = x2 - tx * lx2 * gradient2
                    x1 = np.maximum(np.zeros(tmp1.shape), np.abs(tmp1) - tx) * np.sign(
                        tmp1
                    )
                    x2 = np.maximum(np.zeros(tmp2.shape), np.abs(tmp2) - tx) * np.sign(
                        tmp2
                    )

            cost1 = MinRankKernel.get_cost(x1, x2, _y1, _y2, _k, params)

            if cost1 > 2 * cost0 or np.isnan(cost1):
                tx = tx / 2
                x1 = x1_backup.copy()
                x2 = x2_backup.copy()
            else:
                break
        return x1, x2

    @staticmethod
    def MRK_optimize_kernel(
        _x1: np.ndarray,
        _x2: np.ndarray,
        _kernel: np.ndarray,
        _y1: np.ndarray,
        _y2: np.ndarray,
        _tmpmu: float,
        _params: MinRankKernelParam,
    ):
        """Optimalizatin of kernel.

        Args:
            _x1 (np.ndarray): Input image 1
            _x2 (np.ndarray): Input image 2
            _k (np.ndarray): Kernel
            _y1 (np.ndarray): Output image 1
            _y2 (np.ndarray): Output image 2
            _tmpmu (float): Temporar optimization paramter
            _params (MinRankKernelParam): Algorithm paramters

        Returns:
            (np.ndarray): Output kernel
        """
        kernel_backup = _kernel.copy()
        (gradLS1, gradLS2) = MinRankKernel.get_grad(_x1, _x2, _kernel)
        grad_kernel = gradLS1 + gradLS2 + 2 * _tmpmu * _kernel

        joined_conv = 2 * fftconvolve(np.rot90(_x1, 2), _y1, "valid") + 2 * fftconvolve(
            np.rot90(_x2, 2), _y2, "valid"
        )
        joined_conv = joined_conv + 2 * _tmpmu * kernel_backup
        grad_diff = joined_conv - grad_kernel

        grad_diff_backup = grad_diff.copy()
        e1 = np.matmul(
            np.transpose(np.concatenate(grad_diff)), np.concatenate(grad_diff)
        )
        e0 = e1

        i = 0
        while e1 > _params.ep * e0 and i < _params.opt_k_iter_max:
            (gradLS1, gradLS2) = MinRankKernel.get_grad(_x1, _x2, grad_diff_backup)
            Ad = gradLS1 + gradLS2 + 2 * _tmpmu * grad_diff_backup
            alpha = e1 / (
                np.matmul(
                    np.transpose(np.concatenate(grad_diff_backup)), np.concatenate(Ad)
                )
            )
            _kernel = _kernel + alpha * grad_diff_backup

            if not np.mod(i, 50):
                (gradLS1, gradLS2) = MinRankKernel.get_grad(_x1, _x2, _kernel, _y1, _y2)
                gradLS = gradLS1 + gradLS2 + 2 * _tmpmu * (_kernel - kernel_backup)

                grad_diff = -gradLS
            else:
                grad_diff = grad_diff - alpha * Ad

            e0 = e1
            e1 = np.matmul(
                np.transpose(np.concatenate(grad_diff)), np.concatenate(grad_diff)
            )
            beta = e1 / e0
            grad_diff_backup = grad_diff + beta * grad_diff_backup
            i = i + 1

        return _kernel

    @staticmethod
    def MRK_optimize_rank(
        _kernel: np.ndarray,
        _tau: float,
        _params: MinRankKernelParam,
    ):
        """Optimizes kernel rank to reduce any large kernel effects

        Args:
            _kernel (np.ndarray): Kernel
            _tau (float): Amount of optimalization
            _params (MinRankKernelParam): Algorithm parameters

        Returns:
            np.ndarray: Output rank optimized kernel
        """
        kernel_copy = _kernel.copy()
        weight = _params.mu * np.ones((_kernel.shape[0], 1))

        for _ in range(_params.opt_r_iter_max):
            Uh, Sh, Vh = np.linalg.svd(kernel_copy)
            Lh = np.dot(
                Uh * np.maximum(Sh - (_tau * np.diag(weight)), np.zeros(Sh.shape)), Vh
            )
            SLh = np.linalg.svdvals(Lh)
            weight = np.ones((weight.shape[0], 1)) / (SLh + _params.delta)

        return Lh

    @staticmethod
    def get_grad(
        _x1: np.ndarray,
        _x2: np.ndarray,
        _kernel: np.ndarray,
        _y1=0,
        _y2=0,
    ):
        """Calculates the gradients between images.

        Args:
            _x1 (np.ndarray): Input image 1
            _x2 (np.ndarray): Input image 2
            _kernel (np.ndarray): Kernel
            _y1 (int, optional): Output image 1. Defaults to 0.
            _y2 (int, optional): Output image 2. Defaults to 0.

        Returns:
            (np.ndarray, np.ndarray): Gradiens for both image combinations.
        """
        gradLS1 = 2 * fftconvolve(
            np.rot90(_x1, 2), fftconvolve(_x1, _kernel, "valid") - _y1, "valid"
        )
        gradLS2 = 2 * fftconvolve(
            np.rot90(_x2, 2), fftconvolve(_x2, _kernel, "valid") - _y2, "valid"
        )
        return (gradLS1, gradLS2)

    @staticmethod
    def get_cost(
        _x1: np.ndarray,
        _x2: np.ndarray,
        _y1: np.ndarray,
        _y2: np.ndarray,
        _kernel: np.ndarray,
        params: MinRankKernelParam,
    ):
        """Takes in expected input, output images and kernel and calculates the cost between them.

        Args:
            _x1 (np.ndarray): Input image 1
            _x2 (np.ndarray): Input image 1
            _y1 (np.ndarray): Output image 1
            _y2 (np.ndarray): Output image 2
            _kernel (np.ndarray): Kernel
            params (MinRankKernelParam): Algorihtm parameters

        Returns:
            float: Expected cost
        """
        tmp1 = fftconvolve(_x1, _kernel, "same") - _y1
        tmp2 = fftconvolve(_x2, _kernel, "same") - _y2
        costLS = (
            params.lamb
            / 2
            * (
                np.matmul(np.concatenate(tmp1).T, np.concatenate(tmp1))
                + np.matmul(np.concatenate(tmp2).T, np.concatenate(tmp2))
            )
        )
        costR = norm1(_x1) / norm2(_x1) + norm1(_x2) / norm2(_x2)
        return costLS + costR

    @staticmethod
    def normalize_kernel(_kernel: np.ndarray, _params: MinRankKernelParam):
        """Normalizes a kernel so the sum is exactly 1.

        Args:
            _kernel (np.ndarray): Input kernel
            _params (MinRankKernelParam): Algorithm parameters

        Returns:
            np.ndarray: Normalized kernel
        """
        _kernel[_kernel < np.max(_kernel) * _params.threshold] = 0
        _kernel = _kernel / np.sum(_kernel)
        return _kernel
