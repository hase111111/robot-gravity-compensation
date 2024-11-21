"""provides unit tests for the functions in gravibot"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import unittest
import numpy as np


try:
    from gravibot._robot.link_param import LinkParam
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from gravibot._robot.link_param import LinkParam


class TestRobotLinkParam(unittest.TestCase):
    """test class of gravibot._robot.link_param"""

    def test_link_param_is_readonly(self):
        """when LinkParam is instantiated,
        should return read-only values"""
        link_param = LinkParam(a=1.0, alpha=2.0, d=3.0, theta=np.pi / 2)

        self.assertEqual(link_param.a, 1.0)
        with self.assertRaises(AttributeError):
            link_param.a = "link1"

        self.assertEqual(link_param.alpha, 2.0)
        with self.assertRaises(AttributeError):
            link_param.alpha = 2.0

        self.assertEqual(link_param.d, 3.0)
        with self.assertRaises(AttributeError):
            link_param.d = np.eye(3)

        self.assertEqual(link_param.theta, np.pi / 2)
        with self.assertRaises(AttributeError):
            link_param.theta = np.array([1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
