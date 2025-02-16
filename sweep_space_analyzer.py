"""
Kinectのデータを読み込むためのモジュール
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

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

INITIAL_THETA = [0.0, -0.879, 0.129, -1.559, 0.0, 0.0, 0.0, -0.199]
TARGET_THETA = [-np.pi / 2.7, -1.029, 0.129, -1.559, -0.05, -0.05, 0.0, -0.30]
for idx, theta in enumerate(INITIAL_THETA):
    robot.set_theta(idx, theta)

FILE_NAME = "KinectData.txt"

# kinectのデータは Frame:*, time: *[ms] で区切られている
SEPARATOR = "Frame:"

DATA_NAME = "Born"

SKELETON_NUM = 20

SKELETON_MAP = {
    "SpineBase": 0,
    "SpineMid": 1,
    "Neck": 2,
    "Head": 3,
    "ShoulderLeft": 4,
    "ElbowLeft": 5,
    "WristLeft": 6,
    "HandLeft": 7,
    "ShoulderRight": 8,
    "ElbowRight": 9,
    "WristRight": 10,
    "HandRight": 11,
    "HipLeft": 12,
    "KneeLeft": 13,
    "AnkleLeft": 14,
    "FootLeft": 15,
    "HipRight": 16,
    "KneeRight": 17,
    "AnkleRight": 18,
    "FootRight": 19,
}

# 上半身
UPPER_BODY = [
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
]

SKELETON_CONNECTION = [
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]


class KinectFrameData:
    """
    Kinectのフレームデータを保持するクラス
    """

    def __init__(self, frame, time, *, data_=None):
        self.frame = frame  # Frame number
        self.time = time  # Time in ms
        # data は 3次元座標20個なので，20*3のリスト
        self.data = [0.0, 0.0, 0.0] * 20

        for i in range(20):
            self.data[i * 3] = data_[i * 3]
            self.data[i * 3 + 1] = data_[i * 3 + 1]
            self.data[i * 3 + 2] = data_[i * 3 + 2]

    def __str__(self):
        return f"Frame: {self.frame}, Time: {self.time}, Data: {self.data}"

    def draw(self, ax_: Axes3D):
        """
        3Dグラフに描画する
        """
        for i, j in SKELETON_CONNECTION:
            x = [self.data[i * 3], self.data[j * 3]]
            y = [self.data[i * 3 + 1], self.data[j * 3 + 1]]
            z = [self.data[i * 3 + 2], self.data[j * 3 + 2]]
            ax_.plot(x, y, z, color="green")

        # 上半身の点は赤色で描画，それ以外は緑色で描画
        for s, i in SKELETON_MAP.items():
            # sがUPPER_BODYに含まれている場合は赤色で描画
            if s in UPPER_BODY:
                ax_.scatter(
                    self.data[i * 3],
                    self.data[i * 3 + 1],
                    self.data[i * 3 + 2],
                    color="red",
                )
            else:
                ax_.scatter(
                    self.data[i * 3],
                    self.data[i * 3 + 1],
                    self.data[i * 3 + 2],
                    color="green",
                )


def read_kinect_data(file_name, *, offset=[0, 0, 0]) -> List[KinectFrameData]:
    """
    Kinectのデータを読み込む．
    Kinectのデータは，以下のような形式で保存されている．
    Born等のデータは任意のデータであり，存在しない場合もある．
    Frame:*, time: *[ms]
    Angle: *
    TimeStamp: *
    (arbitrarily) ID: *
    (arbitrarily) Quality: *
    (arbitrarily) Born0: *, *, *
    (arbitrarily) Born1: *, *, *
    (arbitrarily) ...
    (arbitrarily) Born19: *, *, *
    """
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()

    frame_data = []
    cnt = 0

    while cnt < len(lines):
        if SEPARATOR in lines[cnt]:
            # END を含む場合は終了
            if "END" in lines[cnt]:
                break

            # Frame:*, time: *[ms] を前後で分割
            frame_str = lines[cnt].split(",")[0].strip()
            time_str = lines[cnt].split(",")[1].strip().replace("[ms]", "")

            # 文字列から，数字のみを取り出す
            frame = int(frame_str.split(":")[1].strip())
            time = float(time_str.split(":")[1].strip())

            # 2行読み飛ばす
            cnt += 3

            # 次の行が,ID: * であるかどうかを確認
            if "ID" not in lines[cnt]:
                continue

            # さらに2行読み飛ばす
            cnt += 2

            # 20個のデータを読み込む
            res_data = [0.0, 0.0, 0.0] * 20

            for i in range(20):
                born_str = lines[cnt].split(":")[1].strip()
                x, y, z = born_str.split(",")
                res_data[i * 3] = float(x) + float(offset[0])
                res_data[i * 3 + 1] = (float(z) + float(offset[1])) * -1
                res_data[i * 3 + 2] = float(y) + float(offset[2])
                cnt += 1

            frame_data.append(KinectFrameData(frame, time, data_=res_data))

        else:
            cnt += 1

    return frame_data


def draw_table(ax_: Axes3D):
    """
    h=715mm, 730mm * 520mm のテーブルを描画する
    板の厚さは 15mm, 単位はmなので1000分の1倍する
    """
    origin = np.array([0.0, -0.56, 0.0])
    height = 715
    width = 730
    depth = 520

    pos1 = [
        origin[0] - width / 1000 / 2,
        origin[1] - depth / 1000 / 2,
        origin[2] + height / 1000,
    ]

    pos2 = [
        origin[0] + width / 1000 / 2,
        origin[1] - depth / 1000 / 2,
        origin[2] + height / 1000,
    ]

    pos3 = [
        origin[0] + width / 1000 / 2,
        origin[1] + depth / 1000 / 2,
        origin[2] + height / 1000,
    ]

    pos4 = [
        origin[0] - width / 1000 / 2,
        origin[1] + depth / 1000 / 2,
        origin[2] + height / 1000,
    ]

    # 面を貼る
    ax_.plot(
        [pos1[0], pos2[0], pos3[0], pos4[0], pos1[0]],
        [pos1[1], pos2[1], pos3[1], pos4[1], pos1[1]],
        [pos1[2], pos2[2], pos3[2], pos4[2], pos1[2]],
        color="blue",
    )


WORKSPACE_X = [-0.5, 0.5]
WORKSPACE_Y = [-1.0, 0.0]
WORKSPACE_Z = [0.5, 1.5]
GRID_SIZE = 0.05


def compute_sweep_space(data: List[KinectFrameData]):
    """
    Kinectのデータから，人間の掃引空間を計算し，時間で除して確立を求め，3次元配列で返す．
    作業空間は，WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Zで指定され，GRID_SIZEで分割される．
    """
    # 3次元配列を作成
    ret = np.zeros(
        (
            int((WORKSPACE_X[1] - WORKSPACE_X[0]) / GRID_SIZE),
            int((WORKSPACE_Y[1] - WORKSPACE_Y[0]) / GRID_SIZE),
            int((WORKSPACE_Z[1] - WORKSPACE_Z[0]) / GRID_SIZE),
        )
    )

    # 0 ~ 180 までを削除
    data = data[180:]

    # 人間の掃引空間を計算
    for d in data:
        for i in range(20):
            x = d.data[i * 3]
            y = d.data[i * 3 + 1]
            z = d.data[i * 3 + 2]

            if (
                x < WORKSPACE_X[0]
                or x > WORKSPACE_X[1]
                or y < WORKSPACE_Y[0]
                or y > WORKSPACE_Y[1]
                or z < WORKSPACE_Z[0]
                or z > WORKSPACE_Z[1]
            ):
                continue

            # 3次元配列のインデックスを計算
            x_idx = int((x - WORKSPACE_X[0]) / GRID_SIZE)
            y_idx = int((y - WORKSPACE_Y[0]) / GRID_SIZE)
            z_idx = int((z - WORKSPACE_Z[0]) / GRID_SIZE)

            ret[x_idx, y_idx, z_idx] += 1

    # 時間で除して確立を求める
    ret /= float(len(data))

    return ret


if __name__ == "__main__":
    data = read_kinect_data(FILE_NAME, offset=[-0.1, -0.65, 0.75])

    # 作業空間を計算
    sweep_space = compute_sweep_space(data)

    # 作業空間を表示
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    for x in range(sweep_space.shape[0]):
        for y in range(sweep_space.shape[1]):
            for z in range(sweep_space.shape[2]):
                if sweep_space[x, y, z] > 0.0:
                    # 値に応じて色を変える
                    ax.scatter(
                        x * GRID_SIZE + WORKSPACE_X[0],
                        y * GRID_SIZE + WORKSPACE_Y[0],
                        z * GRID_SIZE + WORKSPACE_Z[0],
                        color=plt.cm.hsv(sweep_space[x, y, z] * 255),
                    )

    draw_table(ax)
    robot.draw(ax)
    data[180].draw(ax)

    # 軸のラベルを設定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 3.0)

    plt.show()

    # アニメーションで表示
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # use enumerate
    for i, d in enumerate(data):
        # if i % 10 != 0:
        #     continue
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 3.0)
        # 軸のラベルを設定
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        d.draw(ax)

        robot.draw(ax)

        draw_table(ax)

        plt.pause(0.01)

    plt.show()
