# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np
from dataclasses import dataclass

from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values
from .._math.type import TransMatrix


@dataclass(frozen=True)
class LinkParam:
    a: float
    alpha: float
    d: float
    theta: float
    is_rot_axis: bool = True
    min_val: float = -np.pi
    max_val: float = np.pi

    def get_trans_mat(self) -> TransMatrix:
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )
        return zero_small_values(ans)

    def set_val(self, val: float) -> float:
        if self.is_rot_axis:
            object.__setattr__(self, "theta", val)
        else:
            object.__setattr__(self, "d", val)
        return val
