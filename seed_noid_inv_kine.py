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


robot = gb.Robot(make_robot_param())
LINK_NUM = robot.get_moveable_link_num()

# 初期の関節角度
INITIAL_THETA = [-np.pi / 2.7, -1.029, 0.129, -1.559, -0.05, -0.05, 0.0, -0.25]
INITIAL_DTHETA = [0.0] * LINK_NUM
INITIAL_DDTHETA = [0.0] * LINK_NUM
if LINK_NUM != len(INITIAL_THETA):
    raise ValueError("LINK_NUM must be equal to len(INITIAL_THETA)")

for i in range(LINK_NUM):
    robot.set_theta(i, INITIAL_THETA[i])
INITIAL_POS = robot.get_joint_pos(robot.get_link_num() - 1)
print(f"INITIAL_POS = {INITIAL_POS}")

TARGET_POS = INITIAL_POS - np.array([0.0, 0.0, 0.0001])


def main():
    """メイン関数"""

    # 制御変数
    theta_mx: cs.MX = cs.MX.sym("theta", LINK_NUM)  # type: ignore

    # 制約条件，関節の角度の範囲
    bounds = [None] * LINK_NUM
    for i in range(LINK_NUM):
        bounds[i] = robot.get_moveable_link_bounds(i)

    constraints = []
    # for i in range(LINK_NUM):
    #     constraints.append(theta_mx[i] - bounds[i][0])
    #     constraints.append(bounds[i][1] - theta_mx[i])

    pos = robot.get_joint_pos_casadi(LINK_NUM - 1, theta_mx)

    cost = 100 * cs.sumsqr(pos - TARGET_POS) + cs.sumsqr(
        theta_mx - np.array(INITIAL_THETA)
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
    theta_init = np.array(INITIAL_THETA)

    # ほんのすこし，乱数で初期値をずらす
    theta_init += np.random.rand(LINK_NUM) * 1.0 - 0.5

    opt_result = solver(
        x0=theta_init,
        lbx=-np.pi / 2,
        ubx=np.pi / 2,
        lbg=-0.0,
        ubg=0.0,
    )

    # 最適化結果を取得
    print(f"best theta = {INITIAL_THETA}")
    print(f"theta_opt = {opt_result['x']}")
    print(f"opt_result = {opt_result['f']}")

    res_theta = opt_result["x"]
    for i in range(LINK_NUM):
        robot.set_theta(i, float(res_theta[i]))
    print(f"initial_pos = {INITIAL_POS}")
    print(f"target_pos = {TARGET_POS}")
    print(f"res_pos = {robot.get_joint_pos(robot.get_link_num() - 1)}")

    print(f"diff = {TARGET_POS - INITIAL_POS}")
    print(f"res_diff = {robot.get_joint_pos(robot.get_link_num() - 1) - INITIAL_POS}")


if __name__ == "__main__":
    main()
