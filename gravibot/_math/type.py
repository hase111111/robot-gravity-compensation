"""provides type aliases and functions for type checking"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


TransMatrix: TypeAlias = NDArray[np.float64]
RotationMatrix: TypeAlias = NDArray[np.float64]
PositionVector: TypeAlias = NDArray[np.float64]


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
    Check if the input vector is a position vector(1 x 3).

    Parameters
    ----------
    vector : NDArray
        Input vector.

    Returns
    -------
    bool
        True if the input vector is a position vector (1 x 3).
    """
    if vector.shape == (3,):
        return True
    else:
        return False


def make_pos_vector(x: float, y: float, z: float) -> PositionVector:
    """
    Make a position vector from x, y, and z.

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    z : float
        z coordinate.

    Returns
    -------
    PositionVector
        Position vector.
    """
    return np.array([x, y, z])
