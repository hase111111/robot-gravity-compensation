"""provide functions for converting one matrix to another"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .type import TransMatrix, RotationMatrix, PositionVector
from .type import is_pos_vector, is_rot_matrix, is_trans_matrix


def trans2pos(trans: TransMatrix) -> PositionVector:
    """
    Parameters
    ----------
    trans : TransMatrix
        4x4の同時変換行列．

    Returns
    -------
    pos : PositionVector
        1x3の位置ベクトル．
    """

    # 入力が4x4行列であることを確認
    if is_trans_matrix(trans) is False:
        raise ValueError("Input matrix must be 4x4.")

    return np.array([trans[0][3], trans[1][3], trans[2][3]]).transpose()


def trans2rot(trans: TransMatrix) -> RotationMatrix:
    """
    Parameters
    ----------
    trans : np.ndarray
        4x4の同次変換行列．

    Returns
    -------
    rot : np.ndarray
        3x3の回転行列．
    """

    # 入力が4x4行列であることを確認
    if is_trans_matrix(trans) is False:
        raise ValueError("Input matrix must be 4x4.")

    # 同次変換行列の上3x3部分を回転行列として抽出
    rot = trans[:3, :3]

    return rot


def rot2trans(rot: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    rot : np.ndarray
        3x3の回転行列．
    translation : np.ndarray
        1x3または3x1の平行移動ベクトル．

    Returns
    -------
    trans : np.ndarray
        4x4の同次変換行列．
    """
    # 入力が適切なサイズであることを確認
    if is_rot_matrix(rot) is False or is_pos_vector(translation) is False:
        raise ValueError("Input matrix must be 3x3 and 1x3 or 3x1.")

    # 平行移動ベクトルを1x3に整形
    translation = np.ravel(translation)

    # 4x4の同次変換行列を構築
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = translation

    return trans
