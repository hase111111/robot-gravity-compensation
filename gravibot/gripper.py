import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from .render import draw_cylinder
from ._math import *
from .robot import Robot, RobotParam


class EndEffecter:
    def __init__(self, com_pos=np.zeros(3)):
        self.origin = np.zeros(3)
        self.com_pos = com_pos
        self.mass = 2.0
        self.AXIS_LENGTH = 20

    def draw(self, ax, trans):
        self.origin = trans2pos(trans)
        pos1 = self.origin
        pos2 = [self.com_pos[0] * 2, self.com_pos[1] * 2, self.com_pos[2] * 2]
        pos2 = trans2rot(trans) @ pos2 + pos1

        self.draw_censor_power(trans, ax)
        tmp = (trans2rot(trans) @ [0, 0, 1]) * self.AXIS_LENGTH
        ax.quiver(
            self.origin[0],
            self.origin[1],
            self.origin[2],
            tmp[0],
            tmp[1],
            tmp[2],
            color="red",
        )
        tmp = (trans2rot(trans) @ [0, 1, 0]) * self.AXIS_LENGTH
        ax.quiver(
            self.origin[0],
            self.origin[1],
            self.origin[2],
            tmp[0],
            tmp[1],
            tmp[2],
            color="blue",
        )
        tmp = (trans2rot(trans) @ [1, 0, 0]) * self.AXIS_LENGTH
        ax.quiver(
            self.origin[0],
            self.origin[1],
            self.origin[2],
            tmp[0],
            tmp[1],
            tmp[2],
            color="green",
        )

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
            radius=0.9, height=length, num_slices=20, color="gray", ax=ax, trans=trans
        )

        pin_length = length / 5
        trans1 = (
            trans
            @ get_trans4x4(0.45, 0, (length + pin_length) / 2)
            @ get_rot4x4("z", np.pi / 4)
        )
        trans2 = (
            trans
            @ get_trans4x4(-0.45, 0, (length + pin_length) / 2)
            @ get_rot4x4("z", np.pi / 4)
        )

        draw_cylinder(
            radius=0.3,
            height=pin_length,
            num_slices=4,
            color="black",
            ax=ax,
            trans=trans1,
        )
        draw_cylinder(
            radius=0.3,
            height=pin_length,
            num_slices=4,
            color="black",
            ax=ax,
            trans=trans2,
        )

    def get_censor_power(self, rot_mat: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rot_mat : np.ndarray
            3x3の回転行列．
            これはローカル座標系からグローバル座標系への変換行列である．
        """
        # 重力加速度
        g = 9.8

        # 重力ベクトル
        gravity = np.array([0, 0, -g])

        # 重力ベクトルをローカル座標系に変換
        ans = rot_mat.T @ gravity * self.mass
        return np.array([ans[2], ans[1], ans[0]])

    def draw_censor_power(self, trans, ax):
        rot = trans2rot(trans)
        power = self.get_censor_power(rot)
        pos = self.origin + self.com_pos @ rot.T
        # 力の矢印を描画
        tmp = np.array([0, 0, power[0]]) @ rot.T
        ax.quiver(pos[0], pos[1], pos[2], tmp[0], tmp[1], tmp[2], color="magenta")
        tmp = np.array([0, power[1], 0]) @ rot.T
        ax.quiver(pos[0], pos[1], pos[2], tmp[0], tmp[1], tmp[2], color="cyan")
        tmp = np.array([power[2], 0, 0]) @ rot.T
        ax.quiver(pos[0], pos[1], pos[2], tmp[0], tmp[1], tmp[2], color="lime")


def reset_graph(ax: Axes3D) -> None:
    ax.clear()
    ax.set_xlim(-25, 25)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-25, 25)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0, 50)
    ax.set_zlabel("Z [m]")
    ax.set_aspect("equal")


def main():
    param = RobotParam()
    param.add_link(a=0, alpha=np.pi / 2, d=10, theta=0)
    param.add_link(a=10, alpha=-np.pi / 2, d=0, theta=0)
    param.add_link(a=10, alpha=0, d=0, theta=0)

    robot = Robot(param)
    endeffecter = EndEffecter([3, 0, 0])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    reset_graph(ax)
    robot.draw(ax)

    # スライダーの追加
    sliders = []
    slider_ax_start = 0.13
    for i in range(len(param.link)):
        ax_slider = plt.axes([0.2, slider_ax_start, 0.65, 0.03])
        slider = Slider(ax_slider, f"Link {i+1} θ", -np.pi, np.pi, valinit=0.0)
        sliders.append(slider)
        slider_ax_start -= 0.025

    def update(val):
        reset_graph(ax)
        for i, slider in enumerate(sliders):
            param.set_theta(i, slider.val)
        robot.draw(ax)
        endeffecter.origin = robot.get_joint_pos(2)
        endeffecter.draw(ax, robot.get_joint_trans(2))
        plt.draw()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
