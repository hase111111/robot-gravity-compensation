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
    # waist_y_joint -2.09 ~ 2.09
    ret = gb.RobotParam()
    ret.add_link(
        gb.LinkParam(
            a=0.134202, alpha=np.pi / 2, d=0.251871, min_val=-2.09, max_val=2.09
        )
    )
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    # l_shoulder_pitch_joint -1.55 ~ 0.29
    ret.add_link(
        gb.LinkParam(
            a=0.027177, alpha=-np.pi / 2, d=0.064702, min_val=-1.55, max_val=0.29
        )
    )
    # l_shoulder_roll_joint -0.08 ~ 1.57
    ret.add_link(
        gb.LinkParam(a=-0.0455, alpha=np.pi / 2, d=0.0, min_val=-0.08, max_val=1.57)
    )
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    # l_shoulder_yaw_joint -1.57 ~ 1.57
    ret.add_link(
        gb.LinkParam(a=0.0, alpha=-np.pi / 2, d=-0.2085, min_val=-1.57, max_val=1.57)
    )
    ret.add_link(
        gb.LinkParam(a=0.0, alpha=0.0, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2)
    )
    # l_elbow  -1.50 ~ 0 合わせて3.00
    ret.add_link(gb.LinkParam(a=0.07, alpha=0.0, d=0.0, min_val=-1.50, max_val=0.0))
    ret.add_link(gb.LinkParam(a=0.1095, alpha=0.0, d=0.0, min_val=-1.50, max_val=0.0))
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=-np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    # l_wrist_yaw_joint -1.57 ~ 1.57
    ret.add_link(
        gb.LinkParam(a=0.0, alpha=np.pi / 2, d=-0.124, min_val=-1.57, max_val=1.57)
    )
    ret.add_link(
        gb.LinkParam(
            a=0.0, alpha=-np.pi / 2, d=0.0, min_val=np.pi / 2, max_val=np.pi / 2
        )
    )
    # l_wrist_roll_joint -1.57 ~ 0.34
    ret.add_link(gb.LinkParam(a=-0.15, alpha=0.0, d=0.0, min_val=-1.57, max_val=0.34))

    return ret


origin = [0.0, 0.0, 0.96]
robot = gb.Robot(make_robot_param(), origin=origin)  # type: ignore
LINK_NUM = robot.get_moveable_link_num()

# 初期の関節角度
INITIAL_THETA = [0.0, -0.879, 0.129, -1.559, 0.0, 0.0, 0.0, -0.199]
INITIAL_DTHETA = [0.0] * LINK_NUM
INITIAL_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(INITIAL_THETA):
    raise ValueError("LINK_NUM must be equal to len(INITIAL_THETA)")

for i_ in range(LINK_NUM):
    robot.set_theta(i_, INITIAL_THETA[i_])
INITIAL_POS = robot.get_joint_pos(robot.get_link_num() - 1)
print(f"INITIAL_POS = {INITIAL_POS}")

# 目標の関節角度
TARGET_THETA = [-np.pi / 2.7, -1.029, 0.129, -1.559, -0.05, -0.05, 0.0, -0.60]
TARGET_DTHETA = [0.0] * LINK_NUM
TARGET_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(TARGET_THETA):
    raise ValueError("LINK_NUM must be equal to len(TARGET_THETA)")

for i_ in range(LINK_NUM):
    robot.set_theta(i_, TARGET_THETA[i_])
TARGET_POS = robot.get_joint_pos(robot.get_link_num() - 1)
print(f"TARGET_POS = {TARGET_POS}")

# 作業空間
WORKSPACE_X = [-0.5, 0.5]
WORKSPACE_Y = [-1.0, 0.0]
WORKSPACE_Z = [0.0, 1.0]
GRID_SIZE = 0.1

# 作業空間をグリッドで分割し，危険値を設定(0:安全, 1:危険)
GRID_X = int((WORKSPACE_X[1] - WORKSPACE_X[0]) / GRID_SIZE)
GRID_Y = int((WORKSPACE_Y[1] - WORKSPACE_Y[0]) / GRID_SIZE)
GRID_Z = int((WORKSPACE_Z[1] - WORKSPACE_Z[0]) / GRID_SIZE)

workspace_grid = np.zeros((GRID_X, GRID_Y, GRID_Z))

WARNING_X = [-0.1, 0.1]
WARNING_Y = [-0.75, -0.5]
WARNING_Z = [0.0, 0.5]

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
x_cs = cs.MX.sym("x")
y_cs = cs.MX.sym("y")
z_cs = cs.MX.sym("z")

# インデックスの計算
ix = cs.floor((x_cs - WORKSPACE_X[0]) / GRID_SIZE)
iy = cs.floor((y_cs - WORKSPACE_Y[0]) / GRID_SIZE)
iz = cs.floor((z_cs - WORKSPACE_Z[0]) / GRID_SIZE)

# インデックスの範囲制限
ix = cs.if_else(ix < 0, 0, cs.if_else(ix >= GRID_X, GRID_X - 1, ix))
iy = cs.if_else(iy < 0, 0, cs.if_else(iy >= GRID_Y, GRID_Y - 1, iy))
iz = cs.if_else(iz < 0, 0, cs.if_else(iz >= GRID_Z, GRID_Z - 1, iz))

# グリッド値の取得（if_elseを使って手動で探索）
GRID_VALUE = 0
for i_ in range(GRID_X):
    for j_ in range(GRID_Y):
        for k_ in range(GRID_Z):
            condition = cs.logic_and(cs.logic_and(ix == i_, iy == j_), iz == k_)
            GRID_VALUE = cs.if_else(condition, workspace_grid[i_, j_, k_], GRID_VALUE)

# CasADiの関数として定義
grid_lookup = cs.Function("grid_lookup", [x_cs, y_cs, z_cs], [GRID_VALUE])


# 時間のリスト
END_TIME = 10.0
TIME_STEP = 0.2
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

        # past_pos = None
        for j in range(LINK_NUM):
            pos = robot_.get_joint_pos_casadi(j, now_theta)
            ret += grid_lookup(pos[0], pos[1], pos[2])
            # if past_pos is not None:
            #     center = pos + (pos - past_pos) / 2
            #     ret += grid_lookup(center[0], center[1], center[2])
            #     center = pos + (pos - past_pos) / 4
            #     ret += grid_lookup(center[0], center[1], center[2])
            #     center = pos + (pos - past_pos) * 3 / 4
            #     ret += grid_lookup(center[0], center[1], center[2])
            # past_pos = pos

    return ret


def clamp_result(theta: np.ndarray, robot_: gb.Robot) -> np.ndarray:
    """結果を制約に合わせて修正"""
    ret = theta.copy()
    for i in range(TIME_NUM):
        for j in range(LINK_NUM):
            min_, max_ = robot_.get_moveable_link_bounds(j)
            if ret[j][i] < min_:
                ret[j][i] = min_
            if ret[j][i] > max_:
                ret[j][i] = max_

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
    ax1.set_ylabel("theta [deg]")
    ax1.set_title("theta")

    # 微分して，角速度を求める
    d = np.diff(angle)
    ax2 = fig.add_subplot(312)
    for i in range(LINK_NUM):
        ax2.plot(time_[:-1], d[i])
    ax2.set_ylabel("dtheta [deg/s]")
    ax2.set_title("dtheta")

    # さらに微分して，角加速度を求める
    dd = np.diff(d)
    ax3 = fig.add_subplot(313)
    for i in range(LINK_NUM):
        ax3.plot(time_[:-2], dd[i])
    ax3.set_ylabel("ddtheta [deg/s^2]")
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

    # 中間点の角度を求める
    theta_center = cs.vertcat()
    for i in range(LINK_NUM):
        theta_center = cs.vertcat(theta_center, theta_mx[i * TIME_NUM + TIME_NUM // 2])

    theta_relay_point = [0.3, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]

    # コスト関数を定義
    cost = smooth_objective(ddtheta_mx)  # + constraints_obstacle(theta_mx, robot)

    # 制約条件
    constraints = cs.vertcat(
        theta_first - INITIAL_THETA,
        theta_last - TARGET_THETA,
        dtheta_first - INITIAL_DTHETA,
        dtheta_last - TARGET_DTHETA,
        ddtheta_first - INITIAL_DDTHETA,
        ddtheta_last - TARGET_DDTHETA,
        theta_center - cs.vertcat(*theta_relay_point),
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
    theta_init = [np.random.uniform(-np.pi / 2, np.pi / 2)] * (LINK_NUM * TIME_NUM)
    # theta_init = [0.0] * (LINK_NUM * TIME_NUM)

    opt_result = solver(
        x0=theta_init,
        lbx=-np.pi / 2,
        ubx=np.pi / 2,
        lbg=-0.0,
        ubg=0.0,
    )

    # 最適化結果を取得
    theta_opt = get_result(opt_result["x"], TIME_NUM, LINK_NUM)
    # epsより小さい値を0にする
    theta_opt = np.where(np.abs(theta_opt) < 1e-6, 0.0, theta_opt)
    theta_opt = clamp_result(theta_opt, robot)
    res_str = np.array2string(
        theta_opt,
        separator=", ",
        formatter={"float_kind": "{: .4f}".format},
    )
    print(f"theta_opt = {res_str}")
    print(f"opt_result = {opt_result['f']}")

    # 図に描画
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    ax.set_xlim(-1, 1)
    ax.set_xlabel("X [m]")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Y [m]")
    ax.set_zlim(0, 2)
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

    for _ in range(3):
        for i in range(TIME_NUM):
            ax.clear()

            ax.set_xlim(-1, 1)
            ax.set_xlabel("X [m]")
            ax.set_ylim(-1, 1)
            ax.set_ylabel("Y [m]")
            ax.set_zlim(0, 2)
            ax.set_zlabel("Z [m]")
            ax.set_aspect("equal")

            for j in range(LINK_NUM):
                robot.set_theta(j, theta_opt[j][i])

            draw_obstacle(ax)
            robot.draw(ax)
            plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    main()
