"""provides a sample code to visualize a robot and its end effecter with sliders."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import gravibot as gb

plt.rcParams["font.family"] = "TakaoPGothic"
plt.rcParams["font.size"] = 12


def reset_graph(ax: Axes3D) -> None:
    ax.clear()
    ax.set_xlim(-25, 25)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-25, 25)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0, 50)
    ax.set_zlabel("Z [m]")
    ax.set_aspect("equal")


def reset_table(ax) -> None:
    ax.clear()
    ax.axis("off")


def make_robot_param() -> gb.RobotParam:
    param = gb.RobotParam()
    param.add_link(a=0, alpha=np.pi / 2, d=10, theta=0)
    param.add_link(a=10, alpha=-np.pi / 2, d=0, theta=0)
    param.add_link(a=10, alpha=0, d=0, theta=0)

    return param


def add_slider(ax, slider_ax_start, num_links):
    sliders = []
    for i in range(num_links):
        ax_slider = plt.axes([0.2, slider_ax_start, 0.65, 0.03])
        slider = Slider(ax_slider, f"Link {i+1} θ", -np.pi, np.pi, valinit=0.0)
        sliders.append(slider)
        slider_ax_start -= 0.025

    return sliders


def draw_table(ax, robot: gb.Robot, end_effecter: gb.EndEffecter) -> None:
    # ロボットの手先の位置を取得
    pos = robot.get_joint_pos(2)
    coord = gb.trans2rot(robot.get_joint_trans(2))
    com_pos = [
        end_effecter.com_pos[2],
        end_effecter.com_pos[1],
        end_effecter.com_pos[0],
    ]
    power = end_effecter.get_censor_power(coord)
    moment = np.cross(com_pos, power)
    data = {
        "手先の位置 [m]": "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(*pos),
        "手先からグリッパの重心へ向かうベクトル [m]": "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(
            *com_pos
        ),
        "グリッパの質量 [kg]": "{:.3f}".format(end_effecter.mass),
        "手先の姿勢(回転行列)": "{:.3f}, {:.3f}, {:.3f} \n {:.3f}, {:.3f}, {:.3f} \n {:.3f}, {:.3f}, {:.3f}".format(
            *coord[0], *coord[1], *coord[2]
        ),
        "センサにかかる力 [N]": "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(*power),
        "センサにかかるモーメント [Nm]": "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(*moment),
    }

    # テーブルのデータ作成
    cell_text = [[key, data[key]] for key in data.keys()]
    col_labels = ["Data", "Value"]

    # テーブルの描画
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.5, 0.6],
    )
    ax.axis("off")
    ax.axis("tight")
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 1.5)

    row_to_expand = 4  # 縦幅を大きくする行のインデックス（データ行の場合0から始まる）
    for col in range(2):  # カラム数
        table[(row_to_expand, col)].set_height(0.1)  # 縦幅を大きく設定


def main():
    param = make_robot_param()
    robot = gb.Robot(param)
    endeffecter = gb.EndEffecter([3, 0, 0])

    # 3Dグラフの初期化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    table = fig.add_subplot(1, 2, 2)

    # グラフの初期化
    reset_graph(ax)
    robot.draw(ax)
    endeffecter.draw(ax, robot.get_joint_trans(2))
    draw_table(table, robot, endeffecter)

    # スライダーの追加
    sliders = add_slider(ax, 0.13, len(param.link))

    # スライダーの更新時に呼び出す関数
    def update(_):
        reset_graph(ax)
        reset_table(table)
        for i, slider in enumerate(sliders):
            param.set_theta(i, slider.val)
        robot.draw(ax)
        endeffecter.origin = robot.get_joint_pos(2)
        endeffecter.draw(ax, robot.get_joint_trans(2))
        draw_table(table, robot, endeffecter)
        plt.draw()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
