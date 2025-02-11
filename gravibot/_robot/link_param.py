"""provide LinkParam class"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .._math.type import TransMatrix
from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values4x4
from .._util.type_check import _type_checked


class LinkParam:
    """class for link parameters"""

    def __init__(
        self,
        a: float,
        alpha: float,
        d: float,
        *,
        min_val: float = -np.pi,
        max_val: float = np.pi,
    ):
        self._a = _type_checked(a, float)
        self._alpha = _type_checked(alpha, float)
        self._d = _type_checked(d, float)
        self._min_val = _type_checked(min_val, float)
        self._max_val = _type_checked(max_val, float)

    def get_trans_mat(self, theta) -> TransMatrix:
        """return link's A matrix"""

        theta = _type_checked(theta, float)

        if not self._min_val <= theta <= self._max_val:
            raise ValueError(
                f"theta should be in range [{self._min_val}, {self._max_val}]"
            )

        ans = (
            get_rot4x4("z", theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )
        return zero_small_values4x4(ans)

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
        raise AttributeError("d is read-only")

    @property
    def min_val(self) -> float:
        """getter for min_val, min_val is read-only"""
        return self._min_val

    @min_val.setter
    def min_val(self, _):
        raise AttributeError("min_val is read-only")

    @property
    def max_val(self) -> float:
        """getter for max_val, max_val is read-only"""
        return self._max_val

    @max_val.setter
    def max_val(self, _):
        raise AttributeError("max_val is read-only")
