
import numpy as np

def axis_name_check(axis: str) -> str:
    '''
    Parameters
    ----------
    axis : str
        回転軸名．

    Returns
    -------
    axis : str
        回転軸名．
    '''
    # axisを名寄せ（小文字化し，空白を削除）
    axis = axis.lower()
    axis = axis.replace(' ', '')

    # axisはx, y, zのいずれか
    if axis != 'x' and axis != 'y' and axis != 'z':
        raise ValueError('axis must be x, y or z')
    
    return axis

def get_rot_mat(axis: str, theta: float) -> np.ndarray:
    '''
    Parameters
    ----------
    axis : str
        回転軸．'x', 'y', 'z'のいずれか．
    theta : float
        回転角．単位はラジアン．

    Returns
    -------
    rot_mat : np.ndarray
        4x4の回転行列．
    '''
    a = axis_name_check(axis)
    
    # 回転行列（同時変換行列）の生成
    if a == 'x':
        rot_mat = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]])
    elif a == 'y':
        rot_mat = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                            [0, 1, 0, 0],
                            [-np.sin(theta), 0, np.cos(theta), 0],
                            [0, 0, 0, 1]])
    else:
        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
    return rot_mat

def get_trans_mat(x: float, y: float, z: float) -> np.ndarray:
    '''
    Parameters
    ----------
    x : float
        x軸方向の移動量．
    y : float
        y軸方向の移動量．
    z : float
        z軸方向の移動量．

    Returns
    -------
    trans_mat : np.ndarray
        4x4の同時変換行列．
    '''

    # 移動行列
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def trans2pos(trans: np.ndarray) -> np.ndarray:
    '''
    Parameters
    ----------
    4x4の同時変換行列．

    Returns
    -------
    pos : np.ndarray
        1x3の位置ベクトル．
    '''

    return np.array([trans[0][3],trans[1][3],trans[2][3]]).transpose()

def trans2rot(trans: np.ndarray) -> np.ndarray:
    '''
    Parameters
    ----------
    trans : np.ndarray
        4x4の同次変換行列．

    Returns
    -------
    rot : np.ndarray
        3x3の回転行列．
    '''
    # 入力が4x4行列であることを確認
    if trans.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4.")

    # 同次変換行列の上3x3部分を回転行列として抽出
    rot = trans[:3, :3]

    return rot

def rot2trans(rot: np.ndarray, translation: np.ndarray) -> np.ndarray:
    '''
    Parameters
    ----------
    rot : np.ndarray
        3x3の回転行列．
    translation : np.ndarray
        1x3または3x1の平行移動ベクトル．

    Returns
    -------
    trans : np.ndarray
        4x4の同次変換行列．
    '''
    # 入力が適切なサイズであることを確認
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if translation.shape not in [(3,), (3, 1), (1, 3)]:
        raise ValueError("Translation vector must be of size 3 (1x3 or 3x1).")
    
    # 平行移動ベクトルを1x3に整形
    translation = np.ravel(translation)

    # 4x4の同次変換行列を構築
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = translation

    return trans

