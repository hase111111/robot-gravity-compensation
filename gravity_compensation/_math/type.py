# -*- coding: utf-8 -*-

# Copyright (c) 2023 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

from numpy.typing import NDArray
from typing import TypeAlias

TransMatrix: TypeAlias = NDArray
RotationMatrix: TypeAlias = NDArray
PositionVector: TypeAlias = NDArray


def is_trans_matrix(matrix: NDArray) -> bool:
    """
    Check if the input matrix is a transformation matrix.

    Parameters
    ----------
    matrix : NDArray
        Input matrix.

    Returns
    -------
    bool
        True if the input matrix is a transformation matrix.
    """
    if matrix.shape == (4, 4):
        return True
    else:
        return False


def is_rot_matrix(matrix: NDArray) -> bool:
    """
    Check if the input matrix is a rotation matrix.

    Parameters
    ----------
    matrix : NDArray
        Input matrix.

    Returns
    -------
    bool
        True if the input matrix is a rotation matrix.
    """
    if matrix.shape == (3, 3):
        return True
    else:
        return False


def is_pos_vector(vector: NDArray) -> bool:
    """
    Check if the input vector is a position vector.

    Parameters
    ----------
    vector : NDArray
        Input vector.

    Returns
    -------
    bool
        True if the input vector is a position vector.
    """
    if vector.shape == (3,):
        return True
    else:
        return False
