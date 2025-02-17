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


origin = [0.0, 0.0, 0.5]
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
TARGET_THETA = [-np.pi / 2.7, -1.029, 0.129, -1.559, -0.05, -0.05, 0.0, -0.30]
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
WORKSPACE_Z = [0.5, 1.5]
GRID_SIZE = 0.1

# 作業空間をグリッドで分割し，危険値を設定(0:安全, 1:危険)
GRID_X = int((WORKSPACE_X[1] - WORKSPACE_X[0]) / GRID_SIZE)
GRID_Y = int((WORKSPACE_Y[1] - WORKSPACE_Y[0]) / GRID_SIZE)
GRID_Z = int((WORKSPACE_Z[1] - WORKSPACE_Z[0]) / GRID_SIZE)

import sweep_space_analyzer as ssa

data = ssa.read_kinect_data(ssa.FILE_NAME, offset=[-0.1, -0.65, 0.75])

# 作業空間を計算
sweep_space = ssa.compute_sweep_space(data)
workspace_grid = np.zeros((GRID_X, GRID_Y, GRID_Z))
workspace_grid = sweep_space

WARNING_X = [-0.1, 0.1]
WARNING_Y = [-0.75, -0.5]
WARNING_Z = [0.0, 0.5]


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

    # コスト関数を定義
    cost = 0.00001 * smooth_objective(ddtheta_mx) + constraints_obstacle(
        theta_mx, robot
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
    # theta_init = [np.random.uniform(-np.pi / 2, np.pi / 2)] * (LINK_NUM * TIME_NUM)
    # theta_init = [0.0] * (LINK_NUM * TIME_NUM)
    theta_init = [
        0.0000,
        0.0000,
        0.0000,
        0.0018,
        0.0068,
        0.0160,
        0.0299,
        0.0487,
        0.0723,
        0.1003,
        0.1322,
        0.1670,
        0.2038,
        0.2414,
        0.2785,
        0.3138,
        0.3461,
        0.3738,
        0.3957,
        0.4106,
        0.4173,
        0.4148,
        0.4023,
        0.3792,
        0.3451,
        0.3000,
        0.2441,
        0.1780,
        0.1027,
        0.0193,
        -0.0709,
        -0.1662,
        -0.2649,
        -0.3654,
        -0.4657,
        -0.5641,
        -0.6590,
        -0.7485,
        -0.8314,
        -0.9062,
        -0.9720,
        -1.0280,
        -1.0737,
        -1.1092,
        -1.1349,
        -1.1514,
        -1.1604,
        -1.1636,
        -1.1636,
        -1.1636,
        -0.8790,
        -0.8790,
        -0.8790,
        -0.8765,
        -0.8695,
        -0.8566,
        -0.8368,
        -0.8096,
        -0.7749,
        -0.7328,
        -0.6841,
        -0.6293,
        -0.5697,
        -0.5064,
        -0.4408,
        -0.3745,
        -0.3090,
        -0.2460,
        -0.1870,
        -0.1335,
        -0.0871,
        -0.0491,
        -0.0205,
        -0.0024,
        0.0045,
        0.0000,
        -0.0161,
        -0.0434,
        -0.0813,
        -0.1289,
        -0.1849,
        -0.2479,
        -0.3165,
        -0.3890,
        -0.4637,
        -0.5389,
        -0.6128,
        -0.6840,
        -0.7508,
        -0.8121,
        -0.8665,
        -0.9133,
        -0.9520,
        -0.9822,
        -1.0041,
        -1.0185,
        -1.0262,
        -1.0290,
        -1.0290,
        -1.0290,
        0.1290,
        0.1290,
        0.1290,
        0.1286,
        0.1277,
        0.1259,
        0.1231,
        0.1193,
        0.1144,
        0.1085,
        0.1017,
        0.0940,
        0.0855,
        0.0766,
        0.0672,
        0.0578,
        0.0484,
        0.0392,
        0.0306,
        0.0227,
        0.0157,
        0.0098,
        0.0052,
        0.0019,
        0.0002,
        0.0000,
        0.0014,
        0.0043,
        0.0086,
        0.0142,
        0.0210,
        0.0288,
        0.0374,
        0.0465,
        0.0560,
        0.0656,
        0.0751,
        0.0843,
        0.0929,
        0.1008,
        0.1078,
        0.1139,
        0.1190,
        0.1229,
        0.1258,
        0.1276,
        0.1286,
        0.1290,
        0.1290,
        0.1290,
        -1.5590,
        -1.5590,
        -1.5590,
        -1.5548,
        -1.5430,
        -1.5212,
        -1.4878,
        -1.4419,
        -1.3831,
        -1.3118,
        -1.2289,
        -1.1357,
        -1.0338,
        -0.9254,
        -0.8127,
        -0.6981,
        -0.5843,
        -0.4740,
        -0.3698,
        -0.2742,
        -0.1897,
        -0.1185,
        -0.0626,
        -0.0235,
        -0.0024,
        0.0000,
        -0.0165,
        -0.0514,
        -0.1037,
        -0.1719,
        -0.2542,
        -0.3484,
        -0.4521,
        -0.5625,
        -0.6771,
        -0.7930,
        -0.9076,
        -1.0182,
        -1.1224,
        -1.2180,
        -1.3033,
        -1.3768,
        -1.4375,
        -1.4851,
        -1.5197,
        -1.5423,
        -1.5546,
        -1.5590,
        -1.5590,
        -1.5590,
        0.0000,
        0.0000,
        0.0000,
        -0.0027,
        -0.0101,
        -0.0238,
        -0.0449,
        -0.0740,
        -0.1112,
        -0.1563,
        -0.2088,
        -0.2678,
        -0.3324,
        -0.4013,
        -0.4729,
        -0.5458,
        -0.6184,
        -0.6888,
        -0.7556,
        -0.8171,
        -0.8717,
        -0.9181,
        -0.9549,
        -0.9813,
        -0.9965,
        -1.0000,
        -0.9917,
        -0.9719,
        -0.9411,
        -0.9004,
        -0.8507,
        -0.7936,
        -0.7305,
        -0.6631,
        -0.5930,
        -0.5219,
        -0.4515,
        -0.3835,
        -0.3194,
        -0.2604,
        -0.2079,
        -0.1625,
        -0.1250,
        -0.0957,
        -0.0743,
        -0.0603,
        -0.0527,
        -0.0500,
        -0.0500,
        -0.0500,
        0.0000,
        0.0000,
        0.0000,
        -0.0027,
        -0.0101,
        -0.0238,
        -0.0449,
        -0.0740,
        -0.1112,
        -0.1563,
        -0.2088,
        -0.2678,
        -0.3324,
        -0.4013,
        -0.4729,
        -0.5458,
        -0.6184,
        -0.6888,
        -0.7556,
        -0.8171,
        -0.8717,
        -0.9181,
        -0.9549,
        -0.9813,
        -0.9965,
        -1.0000,
        -0.9917,
        -0.9719,
        -0.9411,
        -0.9004,
        -0.8507,
        -0.7936,
        -0.7305,
        -0.6631,
        -0.5930,
        -0.5219,
        -0.4515,
        -0.3835,
        -0.3194,
        -0.2604,
        -0.2079,
        -0.1625,
        -0.1250,
        -0.0957,
        -0.0743,
        -0.0603,
        -0.0527,
        -0.0500,
        -0.0500,
        -0.0500,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        -0.1990,
        -0.1990,
        -0.1990,
        -0.1981,
        -0.1957,
        -0.1912,
        -0.1843,
        -0.1750,
        -0.1633,
        -0.1493,
        -0.1332,
        -0.1155,
        -0.0966,
        -0.0770,
        -0.0573,
        -0.0380,
        -0.0200,
        -0.0036,
        0.0103,
        0.0213,
        0.0288,
        0.0323,
        0.0314,
        0.0258,
        0.0154,
        0.0000,
        -0.0203,
        -0.0452,
        -0.0744,
        -0.1073,
        -0.1434,
        -0.1820,
        -0.2224,
        -0.2638,
        -0.3054,
        -0.3465,
        -0.3862,
        -0.4238,
        -0.4587,
        -0.4904,
        -0.5183,
        -0.5421,
        -0.5616,
        -0.5768,
        -0.5877,
        -0.5948,
        -0.5986,
        -0.6000,
        -0.6000,
        -0.6000,
    ]

    # 初期変位と，目標変位を設定
    # for i in range(LINK_NUM):
    #     theta_init[i * TIME_NUM] = INITIAL_THETA[i]
    #     theta_init[i * TIME_NUM + TIME_NUM - 1] = TARGET_THETA[i]

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
