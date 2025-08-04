import numpy as np

R = 8.314462618 # universal gas constant

class NasaPolynomial:
    def __init__(self, low_T, high_T, common_T, low_coeffs, high_coeffs):
        self.low_T = low_T
        self.high_T = high_T
        self.common_T = common_T
        self.low_coeffs = low_coeffs
        self.high_coeffs = high_coeffs

    def _select_coeffs(self, T):
        return self.low_coeffs if T <= self.common_T else self.high_coeffs

    def cp(self, T):
        """Heat capacity at constant pressure (J/mol·K)"""
        a = self._select_coeffs(T)
        return R * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)

    def h(self, T):
        """Enthalpy (J/mol)"""
        a = self._select_coeffs(T)
        return R * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)

    def s(self, T):
        """Entropy (J/mol·K)"""
        a = self._select_coeffs(T)
        return R * (a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6])


    def to_dict(self):
        return {
            "low_T": self.low_T,
            "high_T": self.high_T,
            "common_T": self.common_T,
            "low_coeffs": self.low_coeffs,
            "high_coeffs": self.high_coeffs
        }

    def __repr__(self):
        return f"<NASASpecies {self.name}: {self.low_T}K–{self.T_high}K>"
    
