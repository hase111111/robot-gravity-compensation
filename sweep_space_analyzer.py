"""
Kinectのデータを読み込むためのモジュール
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

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
            ax_.plot(x, y, z, color="blue")
        ax_.scatter(self.data[::3], self.data[1::3], self.data[2::3], color="red")


def read_kinect_data(file_name) -> List[KinectFrameData]:
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
                res_data[i * 3] = float(x)
                res_data[i * 3 + 1] = float(z)
                res_data[i * 3 + 2] = float(y)
                cnt += 1

            frame_data.append(KinectFrameData(frame, time, data_=res_data))

        else:
            cnt += 1

    return frame_data


if __name__ == "__main__":
    data = read_kinect_data(FILE_NAME)

    # アニメーションで表示
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # use enumerate
    for i, d in enumerate(data):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # 軸のラベルを設定
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        d.draw(ax)
        plt.pause(0.01)

    plt.show()
