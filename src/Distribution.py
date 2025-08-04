import numpy as np
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.signal import fftconvolve


class LogNormal:

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self.dist = norm(mean, std)

    def rvs(self):
        return np.exp(self.dist.rvs())


class Distribution:

    def __init__(self, points, pdf):
        self.points = points
        self.pdf = pdf / np.trapz(pdf, points)

    def from_cont_dist(base_dist: rv_continuous, npoints: int=10000, th: float=1e-6):
        base_dist = base_dist
        max_v = base_dist.isf(th)
        X = np.linspace(-max_v, max_v, npoints)
        pdf = base_dist.pdf(X)
        return Distribution(X, pdf)
    
    def from_discrete_dist(base_dist: rv_discrete, min_val: int, max_val: int, *args):
        base_dist = base_dist
        X = np.array(range(min_val, max_val + 1))
        pdf = base_dist.pmf(X, *args)
        return Distribution(X, pdf)
    
    @staticmethod
    def add(dist1, dist2):
        """Computes the PDF of the sum of two distributions via convolution"""
        dx = dist1.points[1] - dist1.points[0]
        xmin = dist1.points[0] + dist2.points[0]
        xmax = dist1.points[-1] + dist2.points[-1]
        pdf_result = fftconvolve(dist1.pdf, dist2.pdf, mode='full') * dx
        new_x = np.linspace(xmin, xmax, len(pdf_result))
        return Distribution(new_x, pdf_result)
    
    @classmethod
    def substract(cls, dist1, dist2):
        """Compute the pdf of the difference of two distributions via convolution"""
        dx = dist1.points[1] - dist1.points[0]
        xmin = dist1.points[0] - dist2.points[-1]
        xmax = dist1.points[-1] - dist2.points[0]
        # we flip the PDF to use convolution also for substraction
        pdf_result = fftconvolve(dist1.pdf, dist2.pdf[::-1], mode='full') * dx
        new_x = np.linspace(xmin, xmax, len(pdf_result))
        return Distribution(new_x, pdf_result)
    
    def cdf(self, v: float) -> float:
        """calculate the CDF of the distribution"""
        # find index of closest point
        i = np.argmin([abs(a - v) for a in self.points])
        # calculate integral up to the closest point
        return np.trapz(self.pdf[:i], self.points[:i])
    
    def icdf(self, q: float) -> float:
        """Return the inverse CDF of the distribution"""
        for i, p in enumerate(self.points):
            cdf = np.trapz(self.pdf[:i], self.points[:i])
            if cdf >= q:
                return p
        return p
    
    def ppf(self, v: float) -> float:
        """calculate the PPF (equivalent to PDF) of the distribution"""
        # find index of closest point
        i = np.argmin([abs(a - v) for a in self.points])
        # calculate integral up to the closest point
        return self.pdf[i]
    
    def __add__(self, other):
        if not isinstance(other, Distribution):
            raise ValueError("Can only add distribution to another distribution")
        return self.add(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, Distribution):
            raise ValueError("Can only substract distribution to another distribution")
        return self.substract(self, other)
    
    def __mul__(self, other):
        if not (type(other) is float or type(other) is int):
            raise ValueError("Can only multiply distribution by a scalar value")
        new_x = self.points * other
        new_pdf = self.pdf / abs(other)
        return Distribution(new_x, new_pdf)