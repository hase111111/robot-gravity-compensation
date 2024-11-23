from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import numpy as np

import gravibot._math as _math
import gravibot._robot as _robot
import gravibot._renderer as _renderer
from ._util.type_check import _type_checked


class Robot:
    """class for robot"""

    def __init__(self, param: _robot.RobotParam):
        if not isinstance(param, _robot.RobotParam):
            raise TypeError(f"param must be RobotParam, not {type(param)}")

        self._param = param
        self._origin = _math.make_zero_pos_vector()

    def get_joint_trans(self, i: int) -> _math.TransMatrix:
        """get the transformation matrix of the i-th joint"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        ans = _math.make_identity_trans_matrix()
        for j in range(i + 1):
            ans = ans @ self._param.get_link(j).get_trans_mat()

        return ans

    def get_joint_pos(self, i):
        """get the position of the i-th joint"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        return _math.conv_trans2pos(self.get_joint_trans(i))

    def draw(self, ax: Axes3D):
        for i in range(self._param.get_num_links() - 1):
            _renderer.draw_cylinder3d_by_trans(
                ax,
                1.5,
                3.0,
                self.get_joint_trans(i),
                color="blue",
            )
            self.draw_link(ax, self.get_joint_pos(i), self.get_joint_pos(i + 1))

        self.draw_link(ax, self._origin, self.get_joint_pos(0))
        _renderer.draw_cylinder3d_by_trans(
            radius=3, height=2.0, num_slices=20, color="green", ax=ax
        )

    def draw_link(self, ax: Axes3D, pos1, pos2):
        length = np.linalg.norm(pos2 - pos1)
        pos1to2 = pos2 - pos1
        pos_center = pos1 + pos1to2 / 2

        if length < 1e-10:
            return

        pos1to2 /= np.linalg.norm(pos1to2)
        z_base = np.array([0, 0, 1])

        if np.allclose(pos1to2, z_base):
            rotation_matrix = np.eye(3)
        else:
            x_new = np.cross(z_base, pos1to2)
            x_new /= np.linalg.norm(x_new)
            y_new = np.cross(pos1to2, x_new)
            rotation_matrix = np.array([x_new, y_new, pos1to2]).T

        _renderer.draw_cylinder3d(
            radius=1,
            height=float(length),
            rot=rotation_matrix,
            pos=pos_center,
            num_slices=20,
            color="red",
            ax=ax,
        )

    def _validate_joint_num(self, idx) -> None:
        max_idx = self._param.get_num_links() - 1
        if not 0 <= idx <= max_idx:
            raise ValueError(f"i must be in range [0, num_links{max_idx}]")
