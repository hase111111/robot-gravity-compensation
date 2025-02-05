"""This module provides classes for robot parameters."""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from .link_param import LinkParam
from .link_param_casadi import LinkParamCasadi
from .robot_param import RobotParam
from .robot_param_casadi import RobotParamCasadi

__all__ = [
    "LinkParam",
    "LinkParamCasadi",
    "RobotParam",
    "RobotParamCasadi",
]
