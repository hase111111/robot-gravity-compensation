"""provides the RobotParam class"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from typing import List, Optional

from .link_param_casadi import LinkParamCasadi


class RobotParamCasadi:
    """provides the RobotParam class"""

    def __init__(self, *, param_list: Optional[List[LinkParamCasadi]] = None) -> None:
        if param_list is None:
            param_list = []

        self._link: List[LinkParamCasadi] = param_list

    def add_link(self, link_param: LinkParamCasadi) -> None:
        """add a link to the robot"""

        self._link.append(link_param)

    def set_val(self, idx, theta) -> None:
        """set the i-th link's theta value"""

        self._link[idx].set_val(theta)

    def get_link(self, idx) -> LinkParamCasadi:
        """get the i-th link.
        note that the index starts from 0"""

        return self._link[idx]

    def get_num_links(self) -> int:
        """get the number of links in the robot"""

        return len(self._link)
