"""provides a sample code to visualize a robot and its end effecter with sliders."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from matplotlib.widgets import Slider
import gravibot as gb


def reset_graph(ax: Axes3D) -> None:
    """Reset the 3D graph for animation."""
    ax.clear()
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0.0, 1.0)
    ax.set_zlabel("Z [m]")
    ax.set_aspect("equal")


def reset_table(ax: Axes3D) -> None:
    """Reset the table for animation."""
    ax.clear()
    ax.axis("off")


def make_robot_param() -> gb.RobotParam:
    """Create a robot parameter."""
    ret = gb.RobotParam()
    ret.add_link(gb.LinkParam(a=0.134202, alpha=np.pi / 2, d=0.251871))
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    ret.add_link(gb.LinkParam(a=0.027177, alpha=-np.pi / 2, d=0.064702))
    ret.add_link(gb.LinkParam(a=-0.0455, alpha=np.pi / 2, d=0.0))
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    ret.add_link(gb.LinkParam(a=0.0, alpha=-np.pi / 2, d=-0.2085))
    ret.add_link(
        gb.LinkParam(a=0.0, alpha=0.0, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2)
    )
    ret.add_link(gb.LinkParam(a=0.07, alpha=0.0, d=0.0))
    ret.add_link(gb.LinkParam(a=0.1095, alpha=0.0, d=0.0))
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    ret.add_link(gb.LinkParam(a=0.0, alpha=np.pi / 2, d=0.124))
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=-np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    ret.add_link(gb.LinkParam(a=0.15, alpha=0.0, d=0.0))

    return ret


def add_slider(slider_ax_start, num_links):
    """Add sliders to the plot"""
    sliders = []
    for i in range(num_links):
        ax_slider = plt.axes([0.2, slider_ax_start, 0.65, 0.03])
        slider = Slider(ax_slider, f"Link {i+1} θ", -np.pi, np.pi, valinit=0.0)
        sliders.append(slider)
        slider_ax_start -= 0.025

    return sliders


def draw_table(ax: Axes3D, robot: gb.Robot, end_effecter: gb.EndEffecter) -> None:
    """Draw a table to show the robot's data."""

    # ロボットの手先の位置を取得
    pos = robot.get_joint_pos(2)
    coord = gb.conv_trans2rot(robot.get_joint_trans(2))
    com_pos = gb.make_pos_vector(
        end_effecter.com_pos[2], end_effecter.com_pos[1], end_effecter.com_pos[0]
    )
    power = end_effecter.get_censor_power(coord)
    moment = np.cross(com_pos, power)
    data = {
        "End Effecter Position [m]": f"{gb.posvec_to_str(pos)}",
        "End Effecter To Gripper COM [m]": f"{gb.posvec_to_str(com_pos)}",
        "Gripper Mass [kg]": f"{end_effecter.mass:.3f}",
        "End Effecter Rot": f"{gb.rotmat_to_str(coord)}",
        "Force On Sensor [N]": f"{gb.posvec_to_str(power)}",
        "Moment On Sensor [Nm]": f"{gb.posvec_to_str(moment)}",
    }

    # テーブルのデータ作成
    cell_text = [[key, value] for key, value in data.items()]
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
    table.set_fontsize(10)
    table.scale(1, 1.5)

    row_to_expand = 4  # 縦幅を大きくする行のインデックス（データ行の場合0から始まる）
    for col in range(2):  # カラム数
        table[(row_to_expand, col)].set_height(0.1)  # 縦幅を大きく設定


def main():
    """Main function to visualize a robot and its end effecter with sliders."""
    robot = gb.Robot(make_robot_param(), origin=[0.0, 0.0, 0.0])
    for i in range(robot.get_link_num()):
        robot.set_theta(i, 0.0)
    endeffecter = gb.EndEffecter([3.0, 0.0, 0.0])
    draw_endeffecter = False

    # 3Dグラフの初期化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    table = fig.add_subplot(1, 2, 2)

    # グラフの初期化
    reset_graph(ax)
    robot.draw(ax)
    if draw_endeffecter:
        endeffecter.draw(ax, robot.get_joint_trans(2))

    try:
        draw_table(table, robot, endeffecter)
    except:
        pass

    # スライダーの追加
    sliders = add_slider(0.20, robot.get_link_num())

    # スライダーの更新時に呼び出す関数
    def update(_):
        reset_graph(ax)
        reset_table(table)
        for i, slider in enumerate(sliders):
            robot.set_theta(i, slider.val)
        robot.draw(ax)
        endeffecter.origin = robot.get_joint_pos(2)
        if draw_endeffecter:
            endeffecter.draw(ax, robot.get_joint_trans(2))

        try:
            draw_table(table, robot, endeffecter)
        except:
            pass

        plt.draw()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
