
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trans as tr

def draw_cylinder(
        ax: Axes3D, radius: float, height: float, trans:np.ndarray=None,
        num_slices: int=20, num_stacks: int=2, color: str='blue') -> None:
    """
    指定した座標と向きで円柱を描画する関数（中心を基準に回転）。

    Parameters:
        radius (float): 円柱の半径
        height (float): 円柱の高さ
        trans (np.ndarray): 4x4の同次変換行列（平行移動と回転）
        num_slices (int): 円周方向の分割数
        num_stacks (int): 高さ方向の分割数
        color (str): 円柱全体の色
        ax (Axes3D): 描画対象の3D軸オブジェクト。指定がない場合は新規作成。
    """
    # 同次変換行列が指定されていない場合は単位行列を使用
    if trans is None:
        trans = np.eye(4)

    # 同時変換行列が4*4であることを確認
    if trans.shape != (4, 4):
        raise ValueError('transformation matrix must be 4x4')
    
    # Axes3DオブジェクトがNoneならば例外を返す
    if ax is None:
        raise ValueError('Axes3D object must be specified')
    
    if num_slices < 3:
        raise ValueError('num_slices must be 3 or more')
    
    num_slices += 1  # 端点を含めるため+1

    # θとzの値を生成
    theta = np.linspace(0, 2 * np.pi, num_slices)
    z = np.linspace(0, height, num_stacks)

    # メッシュグリッドを作成
    theta, z = np.meshgrid(theta, z)

    # 円柱の元の座標を計算（円柱側面）
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 上面と下面の元の座標を計算
    theta_cap = np.linspace(0, 2 * np.pi, num_slices)
    x_cap = radius * np.cos(theta_cap)
    y_cap = radius * np.sin(theta_cap)
    z_top = np.full_like(theta_cap, height)  # 上面のz座標
    z_bottom = np.zeros_like(theta_cap)  # 下面のz座標

    # 平行移動して円柱の中心を原点に合わせる
    center_translation = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, -height / 2],
                                    [0, 0, 0, 1]])

    # 全体の変換行列を適用
    full_transform = trans @ center_translation

    # 側面座標に変換を適用
    points_side = np.vstack((x.ravel(), y.ravel(), z.ravel(), np.ones_like(x.ravel())))
    transformed_side = np.dot(full_transform, points_side)
    x, y, z = transformed_side[0].reshape(x.shape), transformed_side[1].reshape(y.shape), transformed_side[2].reshape(z.shape)

    # 上面座標に変換を適用
    points_top = np.vstack((x_cap, y_cap, z_top, np.ones_like(x_cap)))
    transformed_top = np.dot(full_transform, points_top)
    x_cap_top, y_cap_top, z_top = transformed_top[0], transformed_top[1], transformed_top[2]
    epsilon = 1e-6
    x_cap_top += np.random.uniform(-epsilon, epsilon, size=x_cap.shape)
    y_cap_top += np.random.uniform(-epsilon, epsilon, size=y_cap.shape)

    # 下面座標に変換を適用
    points_bottom = np.vstack((x_cap, y_cap, z_bottom, np.ones_like(x_cap)))
    transformed_bottom = np.dot(full_transform, points_bottom)
    x_cap_bottom, y_cap_bottom, z_bottom = transformed_bottom[0], transformed_bottom[1], transformed_bottom[2]
    x_cap_bottom += np.random.uniform(-epsilon, epsilon, size=x_cap.shape)
    y_cap_bottom += np.random.uniform(-epsilon, epsilon, size=y_cap.shape)

    # 円柱の表面をプロット
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.9, color=color, edgecolor='none')

    # 上面と下面をプロット（同じ色）
    ax.plot_trisurf(x_cap_top, y_cap_top, z_top, color=color, alpha=0.9, edgecolor='none')  # 上面
    ax.plot_trisurf(x_cap_bottom, y_cap_bottom, z_bottom, color=color, alpha=0.9, edgecolor='none')  # 下面

def main():
    # 使用例
    transform = tr.get_rot_mat('y', np.pi / 4) @ tr.get_trans_mat(2, 3, 1)

    # 新規作成の軸で描画
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 複数の円柱を同じ軸に描画
    draw_cylinder(radius=2.0, height=4.0, num_slices=3, color='green', trans=transform, ax=ax)
    draw_cylinder(radius=1.0, height=6.0, num_slices=3, color='red', trans=np.eye(4), ax=ax)

    plt.show()

if __name__ == '__main__':
    main()
