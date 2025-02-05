"""provide LinkParam class"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from .._math.trans import get_rot4x4_casadi, get_trans4x4_casadi


class LinkParamCasadi:
    """class for link parameters"""

    def __init__(
        self,
        a,
        alpha,
        d,
        theta,
        *,
        is_rot_axis=True,
    ):
        self._a = a
        self._alpha = alpha
        self._d = d
        self._theta = theta
        self.is_rot_axis = is_rot_axis

    def get_trans_mat(self):
        """return link's A matrix"""
        ans = (
            get_rot4x4_casadi("z", self.theta)
            @ get_trans4x4_casadi(0.0, 0.0, self.d)
            @ get_trans4x4_casadi(self.a, 0.0, 0.0)
            @ get_rot4x4_casadi("x", self.alpha)
        )
        return ans

    def set_val(self, val) -> None:
        """set d and theta"""

        if self.is_rot_axis:
            self._theta = val
        else:
            self._d = val

    @property
    def a(self):
        """getter for a, a is read-only"""
        return self._a

    @a.setter
    def a(self, _):
        raise AttributeError("a is read-only")

    @property
    def alpha(self):
        """getter for alpha, alpha is read-only"""
        return self._alpha

    @alpha.setter
    def alpha(self, _):
        raise AttributeError("alpha is read-only")

    @property
    def d(self):
        """getter for d, d can being changed by set method"""
        return self._d

    @d.setter
    def d(self, _):
        raise AttributeError("Should use set_val method")

    @property
    def theta(self):
        """getter for theta, theta can being changed by set method"""
        return self._theta

    @theta.setter
    def theta(self, _):
        raise AttributeError("Should use set_val method")
