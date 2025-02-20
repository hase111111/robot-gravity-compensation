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


origin = [0.0, 0.0, 0.75]
robot = gb.Robot(make_robot_param(), origin=origin)  # type: ignore
LINK_NUM = robot.get_moveable_link_num()

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

theta_opt = [
    [
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
        -0.5642,
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
        -1.0339,
        -0.9254,
        -0.8127,
        -0.6981,
        -0.5844,
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
        -0.5626,
        -0.6771,
        -0.7931,
        -0.9076,
        -1.0182,
        -1.1224,
        -1.2181,
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
    ],
    [
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
        -0.6889,
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
    ],
    [
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
        -0.6889,
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
    ],
    [
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
    ],
    [
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
        -0.4588,
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
    ],
]

# 危険値を計算
risk_value = sweep_space


def calc_risk_value(theta: np.ndarray, idx) -> float:
    """
    jointの角度から各関節の座標を計算，座標をグリッドに変換し，危険値を取得
    各関節の危険値の合計を返す
    """
    for i in range(LINK_NUM):
        robot.set_theta(i, theta[i][idx])
    risk_sum = 0.0
    for i in range(LINK_NUM):
        pos = robot.get_joint_pos(i)
        grid_x = int((pos[0] - WORKSPACE_X[0]) / GRID_SIZE)
        grid_y = int((pos[1] - WORKSPACE_Y[0]) / GRID_SIZE)
        grid_z = int((pos[2] - WORKSPACE_Z[0]) / GRID_SIZE)
        if grid_x < 0 or grid_x >= GRID_X:
            continue
        if grid_y < 0 or grid_y >= GRID_Y:
            continue
        if grid_z < 0 or grid_z >= GRID_Z:
            continue
        risk_sum += workspace_grid[grid_x, grid_y, grid_z]
    return risk_sum


# 時間-危険値のグラフを描画
risk_value_list = []
for i in range(50):
    risk_value_list.append(calc_risk_value(theta_opt, i))

plt.plot(range(50), risk_value_list, marker="o")
plt.xlabel("time[t]")
plt.ylabel("risk value[J]")

# 0.11 * 10 ^ -6 に線を引く
plt.axhline(y=0.11 * 10**-6, color="r", linestyle="--")

plt.xlim(0, 50)
plt.ylim(0, 1 * 10**-6)

# 凡例を表示
plt.legend(["risk value", "R_danger"])

plt.show()

# ３次元グラフで，ロボットの軌道を描画
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
for i in range(50):
    for j in range(LINK_NUM):
        robot.set_theta(j, theta_opt[j][i])
    robot.draw(ax)

# 危険値が高いところに赤い点を描画
for x in range(GRID_X):
    for y in range(GRID_Y):
        for z in range(GRID_Z):
            if workspace_grid[x, y, z] > 0.0:
                ax.scatter(
                    x * GRID_SIZE + WORKSPACE_X[0],
                    y * GRID_SIZE + WORKSPACE_Y[0],
                    z * GRID_SIZE + WORKSPACE_Z[0],
                    color=plt.cm.hsv(sweep_space[x, y, z] * 255),
                    marker="s",
                    # sizeを変更
                    s=100,
                )

plt.show()
