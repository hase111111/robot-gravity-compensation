from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values
from .._math.type import TransMatrix


class LinkParam:
    def __init__(
        self, a: float, alpha: float, d: float, theta: float, is_rot_axis: bool = True
    ):
        self._a = a
        self._alpha = alpha
        self.d = d
        self.theta = theta
        self._is_rot_axis = is_rot_axis

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, _):
        pass

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, _):
        pass

    @property
    def is_rot_axis(self):
        return self._is_rot_axis

    @is_rot_axis.setter
    def is_rot_axis(self, _):
        pass

    def get_trans_mat(self) -> TransMatrix:
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )

        return zero_small_values(ans)
