# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


def _float_check(val, convert_int_to_float: bool = True) -> float:
    """check if val is float"""
    if convert_int_to_float and isinstance(val, int):
        val = float(val)
    if not isinstance(val, float):
        raise TypeError(f"{val} is not float")
    return val


def _bool_check(val) -> bool:
    """check if val is bool"""
    if not isinstance(val, bool):
        raise TypeError(f"{val} is not bool")
    return val
