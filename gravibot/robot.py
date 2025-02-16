"""Robot class for gravibot"""

from typing import Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

import gravibot._math as _math
import gravibot._robot as _robot
import gravibot._renderer as _renderer
from ._util.type_check import _type_checked


class Robot:
    """class for robot"""

    def __init__(
        self,
        param: _robot.RobotParam,
        *,
        origin: _math.PositionVector = _math.make_zero_pos_vector(),
    ):
        if not isinstance(param, _robot.RobotParam):
            raise TypeError(f"param must be RobotParam, not {type(param)}")

        self._param = param
        self._origin = origin
        self._theta = [0.0] * self._param.get_link_num()
        self._link_radius, self._base_radius = self._get_link_radius()

    def set_theta(self, i: int, theta: float) -> None:
        """set the angle of the i-th joint"""

        i = _type_checked(i, int)
        theta = _type_checked(theta, float)
        self._validate_joint_num(i)
        self._theta[i] = theta

    def get_joint_trans(self, i: int) -> _math.TransMatrix:
        """get the transformation matrix of the i-th joint"""

        if not len(self._theta) == self._param.get_link_num():
            raise ValueError(
                "theta must have the same length as the number of links"
                + f"(theta: {len(self._theta)}, links: {self._param.get_link_num()})"
            )

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        ans = _math.get_trans4x4(*self._origin)
        t_cnt = 0
        for j in range(i + 1):
            if self._param.get_link_param(j).is_fixed():
                ans = ans @ self._param.get_link_param(j).get_trans_mat(
                    self._param.get_link_param(j).min_val
                )
            else:
                ans = ans @ self._param.get_link_param(j).get_trans_mat(
                    self._theta[t_cnt]
                )
                t_cnt += 1

        return ans

    def get_joint_trans_casadi(self, i: int, theta_array_casadi):
        """get the transformation matrix of the i-th joint for casadi"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        ans = _math.get_trans4x4_casadi(*self._origin)
        t_cnt = 0
        for j in range(self.get_link_num()):
            if self._param.get_link_param(j).is_fixed():
                ans = ans @ self._param.get_link_param(j).get_trans_mat_casadi(
                    self._param.get_link_param(j).min_val
                )
            else:
                ans = ans @ self._param.get_link_param(j).get_trans_mat_casadi(
                    theta_array_casadi[t_cnt]
                )
                t_cnt += 1

            if i == t_cnt:
                break

        return ans

    def get_joint_pos(self, i: int) -> _math.PositionVector:
        """get the position of the i-th joint"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        return _math.conv_trans2pos(self.get_joint_trans(i))

    def get_joint_pos_casadi(self, i: int, theta_array_casadi):
        """get the position of the i-th joint for casadi"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        return _math.conv_trans2pos_casadi(
            self.get_joint_trans_casadi(i, theta_array_casadi)
        )

    def get_link_num(self) -> int:
        """get the number of links in the robot"""
        return self._param.get_link_num()

    def get_moveable_link_num(self) -> int:
        """get the number of movable links in the robot"""
        return sum(
            not self._param.get_link_param(i).is_fixed()
            for i in range(self._param.get_link_num())
        )

    def get_moveable_link_bounds(self, ind) -> Tuple[float, float]:
        """get the bounds of the i-th movable link"""
        if not 0 <= ind < self.get_moveable_link_num():
            raise ValueError(f"ind must be in range [0, num_moveable_links{ind}]")

        cnt = 0
        for i in range(self._param.get_link_num()):
            if not self._param.get_link_param(i).is_fixed():
                if cnt == ind:
                    return (
                        self._param.get_link_param(i).min_val,
                        self._param.get_link_param(i).max_val,
                    )
                cnt += 1

        raise ValueError("ind is out of range")

    def draw(self, ax: Axes3D):
        """draw the robot"""
        for i in range(self._param.get_link_num() - 1):
            self._draw_link(ax, self.get_joint_pos(i), self.get_joint_pos(i + 1))

            if (
                i + 1 < self._param.get_link_num()
                and self._param.get_link_param(i + 1).is_fixed()
            ):
                continue

            _renderer.draw_cylinder3d_by_trans(
                ax,
                self._link_radius * 1.2,
                self._link_radius * 3.5,
                self.get_joint_trans(i),
                color="blue",
            )

        self._draw_link(ax, self._origin, self.get_joint_pos(0))

        origin_trans = _math.get_trans4x4(*self._origin)
        _renderer.draw_cylinder3d_by_trans(
            ax,
            self._base_radius,
            self._base_radius * 0.5,
            trans=origin_trans,
            color="green",
        )

    def _draw_link(
        self, ax: Axes3D, pos1: _math.PositionVector, pos2: _math.PositionVector
    ) -> None:
        """draw a link between pos1 and pos2"""

        length = float(np.linalg.norm(pos2 - pos1))
        if length < 1e-10:
            return

        pos1to2: _math.PositionVector = (pos2 - pos1) / length
        pos_center: _math.PositionVector = pos1 + (pos2 - pos1) / 2
        z_base = _math.make_pos_vector(0.0, 0.0, 1.0)

        if np.allclose(pos1to2, z_base):
            rotation_matrix = _math.make_identity_rot_matrix()
        else:
            x_new = np.cross(z_base, pos1to2)
            temp = np.linalg.norm(x_new)
            x_new /= temp if temp > 1e-10 else 1.0
            y_new = np.cross(pos1to2, x_new)
            rotation_matrix = np.array([x_new, y_new, pos1to2]).T

        _renderer.draw_cylinder3d(
            ax,
            self._link_radius,
            length,
            pos_center,
            rotation_matrix,
            color="red",
        )

    def _validate_joint_num(self, idx) -> None:
        max_idx = self._param.get_link_num() - 1
        if not 0 <= idx <= max_idx:
            raise ValueError(f"i must be in range [0, num_links{max_idx}]")

    def _get_link_radius(self) -> Tuple[float, float]:
        # search for the maximum length of the link
        max_length = 0.0
        for i in range(self._param.get_link_num() - 1):
            li_param = self._param.get_link_param(i)
            length = np.sqrt(li_param.a**2 + li_param.d**2)
            max_length = max(max_length, length)

        return max_length / 20, max_length / 10
