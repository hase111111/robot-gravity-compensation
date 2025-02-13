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
    ret.add_link(gb.LinkParam(a=0.0, alpha=np.pi / 2.0, d=10.0, min_val=m, max_val=M))
    ret.add_link(gb.LinkParam(a=10.0, alpha=0.0, d=0.0, min_val=m, max_val=M))
    ret.add_link(gb.LinkParam(a=10.0, alpha=-np.pi / 2.0, d=0.0, min_val=m, max_val=M))
    ret.add_link(gb.LinkParam(a=10.0, alpha=0.0, d=0.0, min_val=m, max_val=M))

    return ret


robot = gb.Robot(make_robot_param())
LINK_NUM = robot.get_link_num()

# 初期の関節角度
INITIAL_THETA = [-cs.pi / 3.0, cs.pi / 5.0, -cs.pi / 5.0 * 2, 0.0]  # 変更可能
INITIAL_DTHETA = [0.0] * LINK_NUM
INITIAL_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(INITIAL_THETA):
    raise ValueError("LINK_NUM must be equal to len(INITIAL_THETA)")

# 目標の関節角度
TARGET_THETA = [cs.pi / 3.0, cs.pi / 5.0, -cs.pi / 5.0 * 2, 0.0]
TARGET_DTHETA = [0.0] * LINK_NUM
TARGET_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(TARGET_THETA):
    raise ValueError("LINK_NUM must be equal to len(TARGET_THETA)")


# 作業空間
WORKSPACE_X = [-30.0, 30.0]
WORKSPACE_Y = [-10.0, 10.0]
WORKSPACE_Z = [0.0, 10.0]
GRID_SIZE = 10.0

# 作業空間をグリッドで分割し，危険値を設定(0:安全, 1:危険)
GRID_X = int((WORKSPACE_X[1] - WORKSPACE_X[0]) / GRID_SIZE)
GRID_Y = int((WORKSPACE_Y[1] - WORKSPACE_Y[0]) / GRID_SIZE)
GRID_Z = int((WORKSPACE_Z[1] - WORKSPACE_Z[0]) / GRID_SIZE)

workspace_grid = np.zeros((GRID_X, GRID_Y, GRID_Z))

WARNING_X = [20.0, 30.0]
WARNING_Y = [-20.0, 20.0]
WARNING_Z = [0.0, 30.0]

for i_ in range(GRID_X):
    for j_ in range(GRID_Y):
        for k_ in range(GRID_Z):
            if (
                WARNING_X[0] <= i_ * GRID_SIZE + WORKSPACE_X[0] < WARNING_X[1]
                and WARNING_Y[0] <= j_ * GRID_SIZE + WORKSPACE_Y[0] < WARNING_Y[1]
                and WARNING_Z[0] <= k_ * GRID_SIZE + WORKSPACE_Z[0] < WARNING_Z[1]
            ):
                workspace_grid[i_, j_, k_] = 1.0


# CasADiのMX変数を定義 (ロボットの位置 x, y, z)
x = cs.MX.sym("x")
y = cs.MX.sym("y")
z = cs.MX.sym("z")

# インデックスの計算
ix = cs.floor((x - WORKSPACE_X[0]) / GRID_SIZE)
iy = cs.floor((y - WORKSPACE_Y[0]) / GRID_SIZE)
iz = cs.floor((z - WORKSPACE_Z[0]) / GRID_SIZE)

# インデックスの範囲制限
ix = cs.if_else(ix < 0, 0, cs.if_else(ix >= GRID_X, GRID_X - 1, ix))
iy = cs.if_else(iy < 0, 0, cs.if_else(iy >= GRID_Y, GRID_Y - 1, iy))
iz = cs.if_else(iz < 0, 0, cs.if_else(iz >= GRID_Z, GRID_Z - 1, iz))

# グリッド値の取得（if_elseを使って手動で探索）
grid_value = 0
for i_ in range(GRID_X):
    for j_ in range(GRID_Y):
        for k_ in range(GRID_Z):
            condition = cs.logic_and(cs.logic_and(ix == i_, iy == j_), iz == k_)
            grid_value = cs.if_else(condition, workspace_grid[i_, j_, k_], grid_value)

# CasADiの関数として定義
grid_lookup = cs.Function("grid_lookup", [x, y, z], [grid_value])


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


def constraints_obstacle(theta: cs.MX, robot_: gb.Robot) -> cs.MX:
    """障害物による制約"""

    ret = 0.0
    for i in range(TIME_NUM):
        now_theta = cs.vertcat()
        for j in range(LINK_NUM):
            now_theta = cs.vertcat(now_theta, theta[j * TIME_NUM + i])

        past_pos = None
        for j in range(LINK_NUM):
            pos = robot_.get_joint_pos_casadi(j, now_theta)
            ret += grid_lookup(pos[0], pos[1], pos[2])
            if past_pos is not None:
                center = pos + (pos - past_pos) / 2
                ret += grid_lookup(center[0], center[1], center[2])
                center = pos + (pos - past_pos) / 4
                ret += grid_lookup(center[0], center[1], center[2])
                center = pos + (pos - past_pos) * 3 / 4
                ret += grid_lookup(center[0], center[1], center[2])
            past_pos = pos

    return ret


def draw_obstacle(ax: Axes3D) -> None:
    """障害物を描画"""
    for i in range(GRID_X):
        for j in range(GRID_Y):
            for k in range(GRID_Z):
                if workspace_grid[i, j, k] == 1:
                    # 四角形を描画
                    x = [
                        i * GRID_SIZE + WORKSPACE_X[0],
                        (i + 1) * GRID_SIZE + WORKSPACE_X[0],
                    ]
                    y = [
                        j * GRID_SIZE + WORKSPACE_Y[0],
                        (j + 1) * GRID_SIZE + WORKSPACE_Y[0],
                    ]
                    z = [
                        k * GRID_SIZE + WORKSPACE_Z[0],
                        (k + 1) * GRID_SIZE + WORKSPACE_Z[0],
                    ]
                    xx, yy = np.meshgrid(x, y)
                    ax.plot_surface(xx, yy, np.full_like(xx, z[0]), alpha=0.5)
                    ax.plot_surface(xx, yy, np.full_like(xx, z[1]), alpha=0.5)

                    y, z = np.meshgrid(y, z)  # type: ignore
                    ax.plot_surface(np.full_like(y, x[0]), y, z, alpha=0.5)
                    ax.plot_surface(np.full_like(y, x[1]), y, z, alpha=0.5)

                    x, z = np.meshgrid(x, z)  # type: ignore
                    ax.plot_surface(x, np.full_like(x, y[0]), z, alpha=0.5)
                    ax.plot_surface(x, np.full_like(x, y[1]), z, alpha=0.5)


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
    cost = 0.0001 * smooth_objective(ddtheta_mx) + constraints_obstacle(theta_mx, robot)

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
    # theta_init = [np.random.uniform(-np.pi, np.pi)] * (LINK_NUM * TIME_NUM)
    theta_init = [0.0] * (LINK_NUM * TIME_NUM)

    # 初期変位と，目標変位を設定
    for i in range(LINK_NUM):
        theta_init[i * TIME_NUM] = INITIAL_THETA[i]
        theta_init[i * TIME_NUM + TIME_NUM - 1] = TARGET_THETA[i]

    opt_result = solver(
        x0=theta_init,
        lbx=-np.pi,
        ubx=np.pi,
        lbg=-0.0,
        ubg=0.0,
    )

    # 最適化結果を取得
    theta_opt = get_result(opt_result["x"], TIME_NUM, LINK_NUM)
    print(f"theta_opt = {theta_opt}")
    print(f"opt_result = {opt_result['f']}")

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
            robot.set_theta(j, theta_opt[j][i * 2])
        robot.draw(ax)

    # 障害物を描画
    draw_obstacle(ax)

    plt.show()

    draw_time_graph(theta_opt * 180 / np.pi, np.arange(0, END_TIME, TIME_STEP))

    # ロボットのアニメーションを描画
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    for _ in range(1):
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
                robot.set_theta(j, theta_opt[j][i])

            draw_obstacle(ax)
            robot.draw(ax)
            plt.pause(0.1)

    plt.show()

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
