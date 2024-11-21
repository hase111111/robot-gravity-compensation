"""provide test cases for gravity_compensation._math.rot"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import unittest
import numpy as np

try:
    from gravity_compensation._math.rot import get_rot3x3
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from gravity_compensation._math.rot import get_rot3x3


class TestMathRot(unittest.TestCase):
    """test class of gravity_compensation._math.rot"""

    def test_get_rot3x3_identity(self):
        """when 0 radian is given,
        should return identity matrix"""
        self.assertTrue((get_rot3x3("x", 0) == np.eye(3)).all())
        self.assertTrue((get_rot3x3("y", 0) == np.eye(3)).all())
        self.assertTrue((get_rot3x3("z", 0) == np.eye(3)).all())

    def test_get_rot3x3_xaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around x axis"""
        self.assertTrue(
            (
                get_rot3x3("x", np.pi / 4)
                == np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                    ]
                )
            ).all()
        )

    def test_get_rot3x3_yaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around y axis"""
        self.assertTrue(
            (
                get_rot3x3("y", np.pi / 4)
                == np.array(
                    [
                        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
                        [0, 1, 0],
                        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)],
                    ]
                )
            ).all()
        )

    def test_get_rot3x3_zaxis(self):
        """when pi/4 radian is given,
        should return rotation matrix around z axis"""
        self.assertTrue(
            (
                get_rot3x3("z", np.pi / 4)
                == np.array(
                    [
                        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                        [0, 0, 1],
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
