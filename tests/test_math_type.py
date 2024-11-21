"""provides unit tests for the functions in gravibot"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import unittest
import numpy as np

try:
    from gravibot._math.type import (
        is_pos_vector,
        is_rot_matrix,
        is_trans_matrix,
        make_pos_vector,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from gravibot._math.type import (
        is_pos_vector,
        is_rot_matrix,
        is_trans_matrix,
        make_pos_vector,
    )


class TestMathTrans(unittest.TestCase):
    """test class of gravibot._math.type"""

    def test_is_pos_vector(self):
        """when a 3x1 vector is given,
        should return True"""
        self.assertTrue(is_pos_vector(np.array([1.0, 2.0, 3.0])))

    def test_is_pos_vector_false(self):
        """when a 3x1 vector is not given,
        should return False"""
        self.assertFalse(is_pos_vector(np.array([1, 2, 3, 4])))
        self.assertFalse(is_pos_vector(np.array([1, 2])))

    def test_is_rot_matrix(self):
        """when a 3x3 matrix is given,
        should return True"""
        self.assertTrue(is_rot_matrix(np.eye(3)))
        self.assertTrue(is_rot_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))

    def test_is_rot_matrix_false(self):
        """when a 3x3 matrix is not given,
        should return False"""
        self.assertFalse(is_rot_matrix(np.eye(2)))
        self.assertFalse(is_rot_matrix(np.eye(4)))
        self.assertFalse(is_rot_matrix(np.array([[1, 0], [0, 1]])))

    def test_is_trans_matrix(self):
        """when a 4x4 matrix is given,
        should return True"""
        self.assertTrue(is_trans_matrix(np.eye(4)))
        self.assertTrue(
            is_trans_matrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            )
        )

    def test_is_trans_matrix_false(self):
        """when a 4x4 matrix is not given,
        should return False"""
        self.assertFalse(is_trans_matrix(np.eye(3)))
        self.assertFalse(is_trans_matrix(np.eye(5)))
        self.assertFalse(
            is_trans_matrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                    ]
                )
            )
        )

    def test_make_pos_vector(self):
        """when a 3x1 vector is given,
        should return a 3x1 vector"""
        self.assertTrue(
            (make_pos_vector(1.0, 2.0, 3.0) == np.array([1.0, 2.0, 3.0])).all()
        )


if __name__ == "__main__":
    unittest.main()
