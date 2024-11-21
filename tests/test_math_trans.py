"""provides unit tests for the functions in gravibot"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import unittest
import numpy as np

try:
    from gravibot._math.trans import get_rot4x4, get_trans4x4, zero_small_values4x4
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from gravibot._math.trans import get_rot4x4, get_trans4x4, zero_small_values4x4


class TestMathTrans(unittest.TestCase):
    """test class of gravibot._math.trans"""

    def test_get_rot4x4_identity(self):
        """when 0 radian is given,
        should return identity matrix"""
        self.assertTrue((get_rot4x4("x", 0) == np.eye(4)).all())
        self.assertTrue((get_rot4x4("y", 0) == np.eye(4)).all())
        self.assertTrue((get_rot4x4("z", 0) == np.eye(4)).all())

    def test_get_rot4x4_xaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around x axis"""
        self.assertTrue(
            (
                get_rot4x4("x", np.pi / 4)
                == np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                        [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                        [0, 0, 0, 1],
                    ]
                )
            ).all()
        )

    def test_get_rot4x4_yaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around y axis"""
        self.assertTrue(
            (
                get_rot4x4("y", np.pi / 4)
                == np.array(
                    [
                        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
                        [0, 1, 0, 0],
                        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4), 0],
                        [0, 0, 0, 1],
                    ]
                )
            ).all()
        )

    def test_get_rot4x4_zaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around z axis"""
        self.assertTrue(
            (
                get_rot4x4("z", np.pi / 4)
                == np.array(
                    [
                        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0],
                        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            ).all()
        )

    def test_get_trans4x4(self):
        """when 1, 2, 3 is given,
        should return translation matrix"""
        self.assertTrue(
            (
                get_trans4x4(2, 3, 4)
                == np.array(
                    [
                        [1, 0, 0, 2],
                        [0, 1, 0, 3],
                        [0, 0, 1, 4],
                        [0, 0, 0, 1],
                    ]
                )
            ).all()
        )

    def test_zero_small_values4x4(self):
        """when small values are given,
        should return matrix that small values are replaced with 0"""
        eps = 1e-10
        self.assertTrue(
            (
                zero_small_values4x4(
                    np.array(
                        [
                            [eps, -eps, eps, -eps],
                            [eps, -eps, eps, -eps],
                            [eps, -eps, eps, -eps],
                            [eps, -eps, eps, -eps],
                        ]
                    )
                )
                == np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
