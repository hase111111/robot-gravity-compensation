"""provide LinkParam class"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .._math.type import TransMatrix
from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values4x4
from .._util.type_check import _float_check, _bool_check


class LinkParam:
    """class for link parameters"""

    def __init__(
        self,
        a: float,
        alpha: float,
        d: float,
        theta: float,
        *,
        is_rot_axis: bool = True,
        min_val: float = -np.pi,
        max_val: float = np.pi,
    ):
        self._a = _float_check(a)
        self._alpha = _float_check(alpha)
        self._d = _float_check(d)
        self._theta = _float_check(theta)
        self.is_rot_axis = _bool_check(is_rot_axis)
        self.min_val = _float_check(min_val)
        self.max_val = _float_check(max_val)

    def get_trans_mat(self) -> TransMatrix:
        """return link's A matrix"""
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )
        return zero_small_values4x4(ans)

    def set_val(self, val: float) -> None:
        """set d and theta"""
        if not self.min_val <= val <= self.max_val:
            raise ValueError(
                f"theta should be in range [{self.min_val}, {self.max_val}]"
            )
        if self.is_rot_axis:
            self._theta = val
        else:
            self._d = val

    @property
    def a(self) -> float:
        """getter for a, a is read-only"""
        return self._a

    @a.setter
    def a(self, _):
        raise AttributeError("a is read-only")

    @property
    def alpha(self) -> float:
        """getter for alpha, alpha is read-only"""
        return self._alpha

    @alpha.setter
    def alpha(self, _):
        raise AttributeError("alpha is read-only")

    @property
    def d(self) -> float:
        """getter for d, d can being changed by set method"""
        return self._d

    @d.setter
    def d(self, _):
        raise AttributeError("Should use set_val method")

    @property
    def theta(self) -> float:
        """getter for theta, theta can being changed by set method"""
        return self._theta

    @theta.setter
    def theta(self, _):
        raise AttributeError("Should use set_val method")
