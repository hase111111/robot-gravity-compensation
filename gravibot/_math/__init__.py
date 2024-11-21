"""This module is __init__.py of gravibot/_math package."""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from .converter import trans2pos, trans2rot, rot2trans
from .rot import get_rot3x3
from .trans import get_rot4x4, get_trans4x4, zero_small_values4x4
from .type import TransMatrix, RotationMatrix
from .type import is_trans_matrix, is_rot_matrix, is_pos_vector

__all__ = [
    "get_rot4x4",
    "get_trans4x4",
    "zero_small_values4x4",
    "get_rot3x3",
    "TransMatrix",
    "RotationMatrix",
    "is_trans_matrix",
    "is_rot_matrix",
    "is_pos_vector",
    "trans2pos",
    "trans2rot",
    "rot2trans",
]
