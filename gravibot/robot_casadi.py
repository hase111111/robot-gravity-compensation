"""Robot class for gravibot"""

import gravibot._math as _math
import gravibot._robot as _robot


class RobotCasadi:
    """class for robot"""

    def __init__(self, param: _robot.RobotParamCasadi):
        if not isinstance(param, _robot.RobotParamCasadi):
            raise TypeError(f"param must be RobotParam, not {type(param)}")

        self._param = param
        self._origin = _math.make_zero_pos_vector_casadi()

    def get_joint_trans(self, i: int):
        """get the transformation matrix of the i-th joint"""

        ans = _math.make_identity_trans_matrix_casadi()
        for j in range(i + 1):
            ans = ans @ self._param.get_link(j).get_trans_mat()

        return ans

    def get_joint_pos(self, i):
        """get the position of the i-th joint"""

        return _math.conv_trans2pos_casadi(self.get_joint_trans(i))
