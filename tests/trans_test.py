import unittest
import numpy as np

try:
    from gravity_compensation._math.trans import *
except:
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import gravity_compensation._math.trans


class TestTransformations(unittest.TestCase):

    def test_axis_name_check(self):
        self.assertEqual(axis_name_check("x"), "x")
        self.assertEqual(axis_name_check("  Y "), "y")
        self.assertEqual(axis_name_check("Z"), "z")
        with self.assertRaises(ValueError):
            axis_name_check("a")

    def test_get_rot4x4(self):
        theta = np.pi / 2
        expected_x = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot4x4("x", theta), expected_x)

        expected_y = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot4x4("y", theta), expected_y)

        expected_z = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot4x4("z", theta), expected_z)

    def test_get_trans4x4(self):
        x, y, z = 1, 2, 3
        expected = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(get_trans4x4(x, y, z), expected)

    def test_get_rot3x3(self):
        theta = np.pi / 2
        expected_x = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot3x3("x", theta), expected_x)

        expected_y = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot3x3("y", theta), expected_y)

        expected_z = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(get_rot3x3("z", theta), expected_z)

    def test_trans2pos(self):
        trans = np.array(
            [
                [1, 0, 0, 4],
                [0, 1, 0, 5],
                [0, 0, 1, 6],
                [0, 0, 0, 1],
            ]
        )
        expected = np.array([4, 5, 6])
        np.testing.assert_array_almost_equal(trans2pos(trans), expected)

    def test_trans2rot(self):
        trans = np.array(
            [
                [1, 0, 0, 4],
                [0, 0, -1, 5],
                [0, 1, 0, 6],
                [0, 0, 0, 1],
            ]
        )
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(trans2rot(trans), expected)

    def test_rot2trans(self):
        rot = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        translation = np.array([4, 5, 6])
        expected = np.array(
            [
                [1, 0, 0, 4],
                [0, 0, -1, 5],
                [0, 1, 0, 6],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(rot2trans(rot, translation), expected)


if __name__ == "__main__":
    unittest.main()
