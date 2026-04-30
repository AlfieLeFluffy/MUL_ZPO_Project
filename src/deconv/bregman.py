import numpy as np
from scipy.signal import fftconvolve

from ..util.image_preprocessing import psf2otf


class Bregman:
    def __init__(self):
        self.lookup_v = 0
        self.xx = 0
        self.known_beta = np.array([])
        self.known_alpha = np.array([])

    def deconvolve(
        self,
        _image: np.ndarray,
        _kernel: np.ndarray,
        _lambda: float,
        _alpha: float,
        _verbose: bool = False,
    ):
        print("--- Starting Bregman ---")
        beta = 400
        initer_max = 5
        outiter_max = 50

        # m = _image.shape[0]
        # n = _image.shape[1]
        km = _kernel.shape[0]
        kn = _kernel.shape[1]
        # ks = int(np.floor(_kernel.shape[0]/2))
        output_image = _image.copy()

        if (np.mod(km, 2) != 1) or (np.mod(kn, 2) != 1):
            raise Exception("Kernel must be odd!")

        dx = np.array([[1, -1]])
        dy = np.array([[1], [-1]])  # np.transpose(dx)
        dxt = np.array([[-1, 1]])  # np.flip(dx,0)
        dyt = np.array([[-1], [1]])  # np.flip(dy,1)

        Ktf, KtK, DtD, Fdx, Fdy = self.compute_constants(_image, _kernel, dx, dy)

        gx = fftconvolve(output_image, dx, "valid")
        gy = fftconvolve(output_image, dy, "valid")

        # fx = convolve2d(_image, dx, 'valid')
        # fy = convolve2d(_image, dy, 'valid')

        # ks = _kernel.shape[0]
        # ks2 = int(np.floor(ks / 2))

        # store some of the statistics
        # lcost = np.array([])
        # pcost = np.array([])
        outiter = 0

        bx = np.zeros(gx.shape)
        by = np.zeros(gy.shape)
        wx = gx.copy()
        wy = gy.copy()

        totiter = 1
        # gk = convolve2d(g, _kernel, 'same')

        # lcost.insert(totiter, (_lambda / 2) * np.pow(ImageProcessor.Deconvolve.norm2(np.concatenate(gk) - np.concatenate(_image)),2))
        # pcost.insert(totiter, np.sum((np.pow(np.abs(np.concatenate(gx)), _alpha))))
        # pcost.insert(totiter, pcost[totiter] + np.sum((np.pow(np.abs(np.concatenate(gy), _alpha)))))

        for outiter in range(outiter_max):
            if _verbose:
                print(f"Outer iteration {outiter + 1}", flush=True)
            else:
                if outiter % 10 == 0:
                    print(f"Bregman iteration {outiter}/ {outiter_max}", flush=True)

            for initer in range(initer_max):
                totiter += 1

                if _alpha == 1:
                    tmpx = beta * (gx + bx)
                    betax = beta
                    tmpx = tmpx / betax

                    tmpy = beta * (gy + by)
                    betay = beta
                    tmpy = tmpy / betay
                    betay = betay
                    wx = np.multiply(
                        np.maximum(
                            np.abs(tmpx) - np.ones(tmpx.shape) / betax,
                            np.zeros(tmpx.shape),
                        ),
                        np.sign(tmpx),
                    )
                    wy = np.multiply(
                        np.maximum(
                            np.abs(tmpy) - np.ones(tmpy.shape) / betay,
                            np.zeros(tmpy.shape),
                        ),
                        np.sign(tmpy),
                    )
                else:
                    wx = self.solve_image_bregman(gx + bx, beta, _alpha)
                    wy = self.solve_image_bregman(gy + by, beta, _alpha)

                bx = bx - wx + gx
                by = by - wy + gy

                wx1 = fftconvolve(wx - bx, dxt, "full")
                wy1 = fftconvolve(wy - by, dyt, "full")
                # tmp = np.zeros(g.shape)

                # gprev = g.copy()
                # gxprev = gx.copy()
                # gyprev = gy.copy()

                num = _lambda * Ktf + beta * np.fft.fft2(wx1 + wy1)
                denom = _lambda * KtK + beta * DtD
                Fg = np.divide(num, denom)
                output_image = np.real(np.fft.ifft2(Fg))

                gx = fftconvolve(output_image, dx, "valid")
                gy = fftconvolve(output_image, dy, "valid")
                # gk = convolve2d(g, _kernel, 'same')
                # lcost.insert(totiter, (_lambda / 2) * np.pow(ImageProcessor.Deconvolve.norm2(np.concatenate(gk) - np.concatenate(_image)),2))
                # pcost.insert(totiter, np.sum((np.pow(np.abs(np.concatenate(gx)), _alpha))))
                # pcost.insert(totiter, pcost[totiter] + np.sum((np.pow(np.abs(np.concatenate(gy), _alpha)))))

        print(f"Bregman iteration {outiter_max}/ {outiter_max}", flush=True)
        print("--- Bregman Finished ---", flush=True)

        return output_image

    def compute_constants(self, _image, _kernel, _dx, _dy):
        sizef = _image.shape
        otfk = psf2otf(_kernel, sizef)
        Ktf = np.conj(otfk) * np.fft.fft2(_image)
        KtK = np.pow(np.abs(otfk), 2)
        Fdx = np.pow(np.abs(psf2otf(_dx, sizef)), 2)
        Fdy = np.pow(np.abs(psf2otf(_dy, sizef)), 2)
        DtD = Fdx + Fdy
        return Ktf, KtK, DtD, Fdx, Fdy

    def solve_image_bregman(self, _input, _beta, _alpha):
        rang = 10
        step = 0.0001

        ind = np.array(
            [np.where(self.known_beta == _beta), np.where(self.known_alpha == _alpha)]
        )

        if self.known_alpha.size == 0 or self.known_beta.size == 0:
            self.xx = np.array(range(-rang, rang, step))

        if np.any(ind):
            print(f"Reusing lookup table for beta {_beta} and alpha {_alpha}")
            # already computed
            w = np.interp1(
                np.transpose(self.xx),
                np.transpose(self.lookup_v[ind, :]),
                np.concatenate(_input),
                "linear",
                "extrap",
            )
            w = np.reshape(w, _input.shape)
        else:
            # now go and recompute xx for new value of beta and alpha
            # tmp = self.compute_w(self.xx, _beta, _alpha)
            pass

    def compute_w(self, _input, _beta, _alpha):
        if np.abs(_alpha - 1) < 1e-9:
            # assume alpha = 1.0
            w = self.compute_w1(_input, _beta)
            return w

        if np.abs(_alpha - 2 / 3) < 1e-9:
            # assume alpha = 2/3
            w = self.compute_w23(_input, _beta)
            return w

        if abs(_alpha - 1 / 2) < 1e-9:
            # assume alpha = 1/2
            w = self.compute_w12(_input, _beta)
            return w

    def compute_w1(self, _input, _beta):
        # solve a simple max problem for alpha = 1
        w = np.maximum(np.abs(_input) - 1 / _beta, np.zeros(_input.shape)) * np.sign(
            _input
        )
        return w

    def compute_w12(self, _input, _beta):
        # solve a cubic equation
        # for alpha = 1/2

        # epsilon = 1e-6
        k = -0.25 / np.pow(_beta, 2)
        m = np.ones(_input.shape) * k * np.sign(_input)

        # t1 = (2 / 3) * _input
        v2 = _input * _input
        v3 = v2 * _input
        t2 = np.exp(
            np.log(
                -27 * m
                - 2 * v3
                + (3 * np.sqrt(3)) * np.sqrt(27 * np.pow(m, 2) + 4 * m * v3)
            )
            / 3
        )
        t3 = v2 / t2
