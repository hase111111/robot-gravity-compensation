# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import unittest

try:
    from gravibot._math.axis import _axis_name_check
except:
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from gravibot._math.axis import _axis_name_check


class TestTransformations(unittest.TestCase):

    def test_axis_name_check1(self):
        # when small x, y, z are given
        # should return x, y, z
        self.assertEqual(_axis_name_check("x"), "x")
        self.assertEqual(_axis_name_check("y"), "y")
        self.assertEqual(_axis_name_check("z"), "z")

    def test_axis_name_check2(self):
        # when large X, Y, Z are given
        # should return x, y, z
        self.assertEqual(_axis_name_check("X"), "x")
        self.assertEqual(_axis_name_check("Y"), "y")
        self.assertEqual(_axis_name_check("Z"), "z")

    def test_axis_name_check3(self):
        # when small x, y, z with space are given
        # should return x, y, z
        self.assertEqual(_axis_name_check(" x "), "x")
        self.assertEqual(_axis_name_check(" y "), "y")
        self.assertEqual(_axis_name_check(" z "), "z")
        self.assertEqual(_axis_name_check(" x"), "x")
        self.assertEqual(_axis_name_check("x "), "x")
        self.assertEqual(_axis_name_check("       x   "), "x")

    def test_axis_name_check4(self):
        # when large X, Y, Z with space are given
        # should return x, y, z
        self.assertEqual(_axis_name_check(" X "), "x")
        self.assertEqual(_axis_name_check(" Y "), "y")
        self.assertEqual(_axis_name_check(" Z "), "z")
        self.assertEqual(_axis_name_check(" X"), "x")
        self.assertEqual(_axis_name_check("X "), "x")
        self.assertEqual(_axis_name_check("       X   "), "x")

    def test_axis_name_check5(self):
        # when invalid axis is given
        # should raise ValueError
        with self.assertRaises(ValueError):
            _axis_name_check("a")
        with self.assertRaises(ValueError):
            _axis_name_check("b")
        with self.assertRaises(ValueError):
            _axis_name_check("c")
        with self.assertRaises(ValueError):
            _axis_name_check("Xx")
        with self.assertRaises(ValueError):
            _axis_name_check("Yy")
        with self.assertRaises(ValueError):
            _axis_name_check("zZ")
        with self.assertRaises(ValueError):
            _axis_name_check("x y")


if __name__ == "__main__":
    unittest.main()
