"""
最適化問題のサンプルプログラム
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

import casadi as cs  # type: ignore

import gravibot as gb


def make_robot_param() -> gb.RobotParam:
    """ロボットのパラメータを作成"""
    m = -np.pi * 2
    M = np.pi * 2
    ret = gb.RobotParam()
    ret.add_link(
        gb.LinkParam(a=0.0, alpha=np.pi / 2.0, d=10.0, theta=0.0, min_val=m, max_val=M)
    )
    ret.add_link(
        gb.LinkParam(a=10.0, alpha=0.0, d=0.0, theta=0.0, min_val=m, max_val=M)
    )
    ret.add_link(
        gb.LinkParam(a=10.0, alpha=-np.pi / 2.0, d=0.0, theta=0.0, min_val=m, max_val=M)
    )
    ret.add_link(
        gb.LinkParam(a=10.0, alpha=0.0, d=0.0, theta=0.0, min_val=m, max_val=M)
    )

    return ret


def make_robot_param_casadi() -> gb.RobotParamCasadi:
    """ロボットのパラメータを作成"""
    ret = gb.RobotParamCasadi()
    ret.add_link(gb.LinkParamCasadi(a=0.0, alpha=np.pi / 2.0, d=10.0, theta=0.0))
    ret.add_link(gb.LinkParamCasadi(a=10.0, alpha=0.0, d=0.0, theta=0.0))
    ret.add_link(gb.LinkParamCasadi(a=10.0, alpha=-np.pi / 2.0, d=0.0, theta=0.0))
    ret.add_link(gb.LinkParamCasadi(a=10.0, alpha=0.0, d=0.0, theta=0.0))

    return ret


param = make_robot_param()
param_casadi = make_robot_param_casadi()
robot = gb.Robot(param)
robot_casadi = gb.RobotCasadi(param_casadi)
LINK_NUM = param.get_link_num()

# 初期の関節角度
INITIAL_THETA = [-cs.pi / 3.0, cs.pi / 5.0, -cs.pi / 5.0 * 2, 0.0]  # 変更可能
INITIAL_DTHETA = [0.0] * LINK_NUM
INITIAL_DDTHETA = [0.0] * LINK_NUM
for i_, t in enumerate(INITIAL_THETA):
    param.set_val(i_, t)

# 目標の関節角度
TARGET_THETA = [cs.pi / 3.0, cs.pi / 5.0, -cs.pi / 5.0 * 2, 0.0]
TARGET_DTHETA = [0.0] * LINK_NUM
TARGET_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(TARGET_THETA):
    raise ValueError("LINK_NUM must be equal to len(TARGET_THETA)")
for i_, t in enumerate(TARGET_THETA):
    param.set_val(i_, t)


# 障害物の位置
OBSTACLE_NUM = 2
OBSTACLE_POS = [gb.make_pos_vector(20.0, 0.0, 0.0), gb.make_pos_vector(20.0, 0.0, 20.0)]
OBSTACLE_RADIUS = [10.0, 7.0]
if OBSTACLE_NUM != len(OBSTACLE_POS) or OBSTACLE_NUM != len(OBSTACLE_RADIUS):
    raise ValueError("OBSTACLE_NUM must be equal to len(OBSTACLE_POS)")

# 時間のリスト
END_TIME = 5.0
TIME_STEP = 0.1
TIME_NUM = int(END_TIME / TIME_STEP)


def get_delta(theta: cs.MX, length: int, dim: int) -> cs.MX:
    """thetaの差分を返す"""
    dtheta = cs.vertcat()
    for i in range(length * dim):
        if i % length != 0:
            dtheta = cs.vertcat(dtheta, theta[i] - theta[i - 1])

    return dtheta


def get_start_data(theta: cs.MX, length: int, dim: int) -> cs.MX:
    """thetaの最初のデータを返す"""
    ret = cs.vertcat()
    for i in range(dim):
        ret = cs.vertcat(ret, theta[i * length])

    return ret


def get_end_data(theta: cs.MX, length: int, dim: int) -> cs.MX:
    """thetaの最後のデータを返す"""
    ret = cs.vertcat()
    for i in range(dim):
        ret = cs.vertcat(ret, theta[i * length + length - 1])

    return ret


def get_result(theta: cs.MX, length: int, dim: int) -> np.ndarray:
    """1次元のデータを間接角度×時間の2次元データに変換"""
    ret = np.zeros((dim, length))
    for i in range(dim):
        for j in range(length):
            ret[i][j] = theta[i * length + j]

    return ret


def smooth_objective(ddtheta: cs.MX) -> float:
    """滑らかさの二乗を返す"""
    jerk = get_delta(ddtheta, TIME_NUM - 2, LINK_NUM)

    # 二乗和を計算
    return cs.sumsqr(jerk)


def constraints_obstacle(
    theta: cs.MX, robot_param_: gb.RobotParamCasadi, robot_: gb.RobotCasadi
):
    """障害物による制約"""
    DIFF = 0.5

    ret = 0.0
    for i in range(OBSTACLE_NUM):
        for j in range(TIME_NUM):
            for k in range(LINK_NUM):
                robot_param_.set_val(k, theta[k * TIME_NUM + j])

            add = 0
            for k in range(1, LINK_NUM):
                pos = robot_.get_joint_pos(k)
                diff = pos - OBSTACLE_POS[i]
                dist = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2  # 距離の二乗
                dist = cs.sqrt(dist)
                add = cs.fmax(add, (OBSTACLE_RADIUS[i] + DIFF) - dist)

            # 障害物の中に入っている場合値を大きくし，外に出ている場合は0
            ret += cs.fmax(0, add)

    return ret


def draw_obstacle(ax: Axes3D) -> None:
    """障害物を描画"""
    for i in range(OBSTACLE_NUM):
        # 球を描画する
        u, v = np.mgrid[0 : (np.pi * 2.0) : 10j, 0 : np.pi : 10j]  # type: ignore
        x = OBSTACLE_RADIUS[i] * np.cos(u) * np.sin(v) + OBSTACLE_POS[i][0]
        y = OBSTACLE_RADIUS[i] * np.sin(u) * np.sin(v) + OBSTACLE_POS[i][1]
        z = OBSTACLE_RADIUS[i] * np.cos(v) + OBSTACLE_POS[i][2]
        ax.plot_surface(x, y, z, color="black", alpha=0.5)


def draw_time_graph(angle: np.ndarray, time_: np.ndarray) -> None:
    """角度，角速度，角加速度と時間のグラフを描画（各関節ごとのデータを全3枚で収める）"""
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    print(f"v.shape = {angle.shape}")
    for i in range(LINK_NUM):
        ax1.plot(time_, angle[i])
    ax1.set_ylabel("theta [rad]")
    ax1.set_title("theta")

    # 微分して，角速度を求める
    d = np.diff(angle)
    ax2 = fig.add_subplot(312)
    for i in range(LINK_NUM):
        ax2.plot(time_[:-1], d[i])
    ax2.set_ylabel("dtheta [rad/s]")
    ax2.set_title("dtheta")

    # さらに微分して，角加速度を求める
    dd = np.diff(d)
    ax3 = fig.add_subplot(313)
    for i in range(LINK_NUM):
        ax3.plot(time_[:-2], dd[i])
    ax3.set_ylabel("ddtheta [rad/s^2]")
    ax3.set_title("ddtheta")

    plt.show()


def main():
    """メイン関数"""

    # 制御変数
    theta_mx: cs.MX = cs.MX.sym("theta", LINK_NUM * TIME_NUM)  # type: ignore
    dtheta_mx = get_delta(theta_mx, TIME_NUM, LINK_NUM)
    ddtheta_mx = get_delta(dtheta_mx, TIME_NUM - 1, LINK_NUM)

    theta_first = get_start_data(theta_mx, TIME_NUM, LINK_NUM)
    theta_last = get_end_data(theta_mx, TIME_NUM, LINK_NUM)
    dtheta_first = get_start_data(dtheta_mx, TIME_NUM - 1, LINK_NUM)
    dtheta_last = get_end_data(dtheta_mx, TIME_NUM - 1, LINK_NUM)
    ddtheta_first = get_start_data(ddtheta_mx, TIME_NUM - 2, LINK_NUM)
    ddtheta_last = get_end_data(ddtheta_mx, TIME_NUM - 2, LINK_NUM)

    # コスト関数を定義
    cost = smooth_objective(ddtheta_mx) + 0.001 * constraints_obstacle(
        theta_mx, param_casadi, robot_casadi
    )

    # 制約条件
    constraints = cs.vertcat(
        theta_first - INITIAL_THETA,
        theta_last - TARGET_THETA,
        dtheta_first - INITIAL_DTHETA,
        dtheta_last - TARGET_DTHETA,
        ddtheta_first - INITIAL_DDTHETA,
        ddtheta_last - TARGET_DDTHETA,
    )

    print(
        f"constraints.shape = {constraints.shape}, is_dense = {constraints.is_dense()}"
    )

    # 最適化問題を定義
    nlp = {
        "x": theta_mx,
        "f": cost,
        "g": constraints,
    }

    # 最適化問題を解く
    solver = cs.nlpsol("solver", "ipopt", nlp)

    # 初期値を設定
    theta_init = [np.random.uniform(-np.pi, np.pi)] * (LINK_NUM * TIME_NUM)

    opt_result = solver(
        x0=theta_init,
        lbx=-np.pi * 2,
        ubx=np.pi * 2,
        lbg=-0.0,
        ubg=0.0,
    )

    # 最適化結果を取得
    theta_opt = get_result(opt_result["x"], TIME_NUM, LINK_NUM)
    print(f"theta_opt = {theta_opt}")

    # 図に描画
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    ax.set_xlim(-25, 25)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-25, 25)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0, 50)
    ax.set_zlabel("Z [m]")
    ax.set_aspect("equal")

    # ロボットを描画
    for i in range(int(TIME_NUM / 2)):
        for j in range(LINK_NUM):
            param.set_val(j, theta_opt[j][i * 2])
        robot.draw(ax)

    # 障害物を描画
    draw_obstacle(ax)

    plt.show()

    draw_time_graph(theta_opt * 180 / np.pi, np.arange(0, END_TIME, TIME_STEP))

    # ロボットのアニメーションを描画
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    for _ in range(10):
        for i in range(TIME_NUM):
            ax.clear()

            ax.set_xlim(-25, 25)
            ax.set_xlabel("X [m]")
            ax.set_ylim(-25, 25)
            ax.set_ylabel("Y [m]")
            ax.set_zlim(0, 50)
            ax.set_zlabel("Z [m]")
            ax.set_aspect("equal")

            for j in range(LINK_NUM):
                param.set_val(j, theta_opt[j][i])

            draw_obstacle(ax)
            robot.draw(ax)
            plt.pause(0.1)

    plt.show()

    exit()

    # # 関節空間の軌跡を描画，縦軸がangle1,横軸がangle2,高さがangle3
    # fig = plt.figure()
    # ax: Axes3D = fig.add_subplot(111, projection="3d")

    # for i in range(TIME_NUM):
    #     ax.scatter(theta_opt[0][i], theta_opt[1][i], theta_opt[2][i])

    # # 全域に対して，障害物と接触する点に赤い点を描画
    # DIV = 15
    # for i in range(DIV):
    #     for j in range(DIV):
    #         for k in range(DIV):
    #             theta = [
    #                 -np.pi + np.pi * 2 * i / DIV,
    #                 -np.pi + np.pi * 2 * j / DIV,
    #                 -np.pi + np.pi * 2 * k / DIV,
    #             ]
    #             for l in range(LINK_NUM):
    #                 param.set_val(l, theta[l])
    #             pos = robot.get_joint_pos(LINK_NUM - 1)
    #             for m in range(OBSTACLE_NUM):
    #                 diff = pos - OBSTACLE_POS[m]
    #                 dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    #                 if dist < OBSTACLE_RADIUS[m]:
    #                     ax.scatter(theta[0], theta[1], theta[2], color="red", s=30)

    # ax.set_xlabel("angle1 [rad]")
    # ax.set_ylabel("angle2 [rad]")
    # ax.set_zlabel("angle3 [rad]")
    # ax.set_title("Joint Space")

    # plt.show()


if __name__ == "__main__":
    main()
