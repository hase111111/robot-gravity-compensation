# -*- coding: utf-8 -*-

# Copyright (c) 2023 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values
from .._math.type import TransMatrix


class LinkParam:
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
        self._a = a
        self._alpha = alpha
        self.d = d
        self.theta = theta
        self._is_rot_axis = is_rot_axis
        self._min_val = min_val
        self._max_val = max_val

    def get_trans_mat(self) -> TransMatrix:
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )

        return zero_small_values(ans)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, _):
        pass

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, _):
        pass

    @property
    def is_rot_axis(self):
        return self._is_rot_axis

    @is_rot_axis.setter
    def is_rot_axis(self, _):
        pass

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, _):
        pass

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, _):
        pass
