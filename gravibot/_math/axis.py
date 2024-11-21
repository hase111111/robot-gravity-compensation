# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


def _axis_name_check(axis: str) -> str:
    """
    入力された回転軸名が正しいか確認する関数．
    x, y, zのいずれかであることを確認する．

    Parameters
    ----------
    axis : str
        回転軸名．

    Returns
    -------
    axis : str
        回転軸名．
    """
    # axisを名寄せ（小文字化し，空白を削除）
    axis = axis.lower()
    axis = axis.replace(" ", "")

    # axisはx, y, zのいずれか
    if axis != "x" and axis != "y" and axis != "z":
        raise ValueError("axis must be x, y or z")

    return axis
