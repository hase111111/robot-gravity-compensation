# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

# from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values
from .._math import get_rot4x4, get_trans4x4, zero_small_values
from .._math.type import TransMatrix


def define_property(self, name, *, init_value=None, can_get=True, can_set=True):
    # "_User__name" のような name mangling 後の名前.
    field_name = "_{}__{}".format(self.__class__.__name__, name)

    # 初期値を設定する.
    setattr(self, field_name, init_value)

    # getter/setter を生成し, プロパティを定義する.
    getter = (lambda self: getattr(self, field_name)) if can_get else None
    setter = (lambda self, value: setattr(self, field_name, value)) if can_set else None
    setattr(self.__class__, name, property(getter, setter))


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
        max_val: float = np.pi
    ):
        define_property(self, "a", init_value=a, can_set=False)
        define_property(self, "alpha", init_value=alpha, can_set=False)

        define_property(self, "is_rot_axis", init_value=is_rot_axis, can_set=False)
        define_property(self, "min_val", init_value=min_val, can_set=False)
        define_property(self, "max_val", init_value=max_val, can_set=False)

        if is_rot_axis:
            define_property(self, "d", init_value=d, can_set=False)
            define_property(self, "theta", init_value=theta)
        else:
            define_property(self, "d", init_value=d)
            define_property(self, "theta", init_value=theta, can_set=False)

    def get_trans_mat(self) -> TransMatrix:
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )
        return zero_small_values(ans)
