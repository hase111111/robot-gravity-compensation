# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .type import TransMatrix, is_trans_matrix
from .axis import _axis_name_check


def get_rot4x4(axis: str, theta: float) -> TransMatrix:
    """
    指定された1軸周りの同時変換行列を生成する関数．
    同時変換行列は4x4の行列である

    Parameters
    ----------
    axis : str
        回転軸．'x', 'y', 'z'のいずれか．
    theta : float
        回転角．単位はラジアン．

    Returns
    -------
    rot_mat : TransMatrix
        4x4の同時変換行列．
    """
    a: str = _axis_name_check(axis)

    # 回転行列（同時変換行列）の生成
    if a == "x":
        rot_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    elif a == "y":
        rot_mat = np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    return rot_mat


def get_trans4x4(x: float, y: float, z: float) -> TransMatrix:
    """
    指定された方向の移動する同時変換行列を生成する関数．

    Parameters
    ----------
    x : float
        x軸方向の移動量 [m].
    y : float
        y軸方向の移動量 [m].
    z : float
        z軸方向の移動量 [m].

    Returns
    -------
    trans_mat : TransMatrix
        4x4の同時変換行列．
    """

    # 移動行列
    return np.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def zero_small_values4x4(trans: TransMatrix) -> TransMatrix:
    """
    小さな値を0に置き換える関数。

    Parameters
    ----------
    trans : TransMatrix
        4x4の同次変換行列。

    Returns
    -------
    TransMatrix
        修正された同次変換行列。
    """
    trans = trans.copy()  # 元の行列を保持するためにコピーを作成

    if is_trans_matrix(trans) is False:
        # 入力が4x4行列でなければ例外を投げる
        raise ValueError("Input matrix must be 4x4.")

    eps: float = 1e-10
    trans[np.abs(trans) <= eps] = 0.0  # 小さな値をゼロに置き換え
    return trans


def trans2pos(trans: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    4x4の同時変換行列．

    Returns
    -------
    pos : np.ndarray
        1x3の位置ベクトル．
    """

    return np.array([trans[0][3], trans[1][3], trans[2][3]]).transpose()


def trans2rot(trans: np.ndarray) -> np.ndarray:
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
    if trans.shape != (4, 4):
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
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if translation.shape not in [(3,), (3, 1), (1, 3)]:
        raise ValueError("Translation vector must be of size 3 (1x3 or 3x1).")

    # 平行移動ベクトルを1x3に整形
    translation = np.ravel(translation)

    # 4x4の同次変換行列を構築
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = translation

    return trans
