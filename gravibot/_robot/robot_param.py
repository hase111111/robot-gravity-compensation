"""provides the RobotParam class"""

from .link_param import LinkParam


class RobotParam:
    """provides the RobotParam class"""

    def __init__(self):
        self._link = []

    def add_link(self, link_param: LinkParam) -> None:
        """add a link to the robot"""

        # check if the link is an instance of LinkParam
        if not isinstance(link_param, LinkParam):
            raise TypeError("link_param must be an instance of LinkParam")

        self._link.append(link_param)

    def set_val(self, i, theta):
        self._link[i].set_val(i, theta)

    def get_link(self, i: int) -> LinkParam:
        """get the i-th link.
        note that the index starts from 0"""

        if i < 0 or i >= len(self._link):
            raise IndexError("index out of range")

        return self._link[i]

    def get_num_links(self) -> int:
        """get the number of links in the robot"""

        return len(self._link)
