import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np
from .render import *
from ._math.trans import *


class LinkParam:
    def __init__(self, a: float, alpha: float, d: float, theta: float):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta

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

    def get_trans_mat(self):
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )

        return zero_small_values(ans)


class RobotParam:
    def __init__(self):
        self.link = []

    def add_link(self, a, alpha, d, theta):
        self.link.append(LinkParam(a, alpha, d, theta))

    def set_theta(self, i, theta):
        self.link[i].theta = theta

    def get_link(self, i):
        return self.link[i]


class Robot:
    def __init__(self, param):
        self._param = param
        self._origin = np.zeros(3)

    def get_joint_trans(self, i):
        trans = np.eye(4)
        for j in range(i + 1):
            trans = trans @ self._param.get_link(j).get_trans_mat()
        return trans

    def get_joint_pos(self, i):
        return trans2pos(self.get_joint_trans(i))

    def draw(self, ax):
        for i in range(len(self._param.link) - 1):
            draw_cylinder(
                radius=1.5,
                height=3.0,
                num_slices=20,
                color="blue",
                ax=ax,
                trans=self.get_joint_trans(i),
            )
            self.draw_link(ax, self.get_joint_pos(i), self.get_joint_pos(i + 1))

        self.draw_link(ax, self._origin, self.get_joint_pos(0))
        draw_cylinder(radius=3, height=2.0, num_slices=20, color="green", ax=ax)

    def draw_link(self, ax, pos1, pos2):
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

        trans = np.array(
            [
                [
                    rotation_matrix[0][0],
                    rotation_matrix[0][1],
                    rotation_matrix[0][2],
                    pos_center[0],
                ],
                [
                    rotation_matrix[1][0],
                    rotation_matrix[1][1],
                    rotation_matrix[1][2],
                    pos_center[1],
                ],
                [
                    rotation_matrix[2][0],
                    rotation_matrix[2][1],
                    rotation_matrix[2][2],
                    pos_center[2],
                ],
                [0, 0, 0, 1],
            ]
        )

        draw_cylinder(
            radius=1, height=length, num_slices=20, color="red", ax=ax, trans=trans
        )


def main():
    param = RobotParam()
    param.add_link(a=0, alpha=np.pi / 2, d=10, theta=0)
    param.add_link(a=10, alpha=0, d=0, theta=0)
    param.add_link(a=10, alpha=0, d=0, theta=0)

    robot = Robot(param)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-30, 30)
    ax.set_aspect("equal")
    robot.draw(ax)

    # スライダーの追加
    sliders = []
    slider_ax_start = 0.25
    for i in range(len(param.link)):
        ax_slider = plt.axes([0.2, slider_ax_start, 0.65, 0.03])
        slider = Slider(ax_slider, f"Link {i+1} θ", -np.pi, np.pi, valinit=0.0)
        sliders.append(slider)
        slider_ax_start -= 0.05

    def update(val):
        ax.clear()
        for i, slider in enumerate(sliders):
            param.set_theta(i, slider.val)
        robot.draw(ax)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)
        ax.set_aspect("equal")
        plt.draw()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
