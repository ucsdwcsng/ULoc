import numpy as np


def get_vr2b_rotation(ap_loc, tag_0_deg_loc):
    PA = np.array(ap_loc)
    PO = np.array(tag_0_deg_loc)
    facing_v = PA - PO
    alpha = np.arctan2(facing_v[2], facing_v[0])

    rotation_vr2b = np.array([
        [0,              1, 0],
        [-np.sin(alpha), 0, np.cos(alpha)],
        [np.cos(alpha),  0, np.sin(alpha)]
    ])
    return rotation_vr2b


def get_b2vr_rotation(ap_loc, tag_0_deg_loc):
    return get_vr2b_rotation(ap_loc, tag_0_deg_loc).T


def get_vr2b_translation(ap_loc, tag_0_deg_loc):
    rotation_vr2b = get_vr2b_rotation(ap_loc, tag_0_deg_loc)
    translation = - rotation_vr2b @ np.array(ap_loc)
    return translation


def get_b2vr_translation(ap_loc, tag_0_deg_loc):
    return np.array(ap_loc)


def get_vr2b_transformation(ap_loc, tag_0_deg_loc):
    r = get_vr2b_rotation(ap_loc, tag_0_deg_loc)
    d = get_vr2b_translation(ap_loc, tag_0_deg_loc)
    # Make 4 x 4 homogeneous transformation matrix
    return np.vstack((np.hstack((r, d[:, None])), [0, 0, 0, 1]))


def get_b2vr_transformation(ap_loc, tag_0_deg_loc):
    r = get_b2vr_rotation(ap_loc, tag_0_deg_loc)
    d = get_b2vr_translation(ap_loc, tag_0_deg_loc)
    # Make 4 x 4 homogeneous transformation matrix
    return np.vstack((np.hstack((r, d[:, None])), [0, 0, 0, 1]))


def vr2b(ap_loc, tag_0_deg_loc, xyz_vr):
    """
    Convert XYZ on VR Axis into AP Board Axis
    :param ap_loc: AP location in VR Axis
    :param tag_0_deg_loc: 0 degree tag location (or any point on normal vector of AP PCB board plane) in VR Axis
    :param xyz_vr: X, Y, Z in VR to be converted to Board Axis
    :return: Converted [X_b, Y_b, Z_b]
    """
    r = get_vr2b_rotation(ap_loc, tag_0_deg_loc)
    t = get_vr2b_translation(ap_loc, tag_0_deg_loc)
    x_b, y_b, z_b = r @ np.array(xyz_vr) + t
    return np.array([x_b, y_b, z_b])


def b2vr(ap_loc, tag_0_deg_loc, xyz_b):
    """
    Convert XYZ on AP Board Axis into VR Global Axis
    :param ap_loc: AP location in VR Axis
    :param tag_0_deg_loc: 0 degree tag location (or any point on normal vector of AP PCB board plane) in VR Axis
    :param xyz_b: X, Y, Z in AP Board Axis to be converted to Board Axis
    :return: Converted [X_vr, Y_vr, Z_vr]
    """
    r = get_b2vr_rotation(ap_loc, tag_0_deg_loc)
    t = get_b2vr_translation(ap_loc, tag_0_deg_loc)
    x_vr, y_vr, z_vr = r @ np.array(xyz_b) + t
    return np.array([x_vr, y_vr, z_vr])


def vr2gtaoa(ap_loc, tag_0_deg_loc, xyz_vr, round_digit=-1):
    assert isinstance(xyz_vr, np.ndarray)
    assert xyz_vr.shape == (3,)
    x_b, y_b, z_b = vr2b(ap_loc, tag_0_deg_loc, xyz_vr)

    theta = -np.arctan(y_b / z_b) * (180 / np.pi)
    phi = -np.arctan(x_b / np.sqrt(y_b ** 2 + z_b ** 2)) * (180 / np.pi)
    if round_digit != -1:
        theta = round(theta, round_digit)
        phi = round(phi, round_digit)
    return theta, phi


if __name__ == '__main__':
    # For testing
    ap_loc = (-1.074, 0.225, 1.540)  # In VR Axis
    tag_0deg_loc = (1.428576, -0.080401, 1.540)  # In VR Axis
    xyz_vr = (1, 22, 0.54)
    test_board = vr2b(ap_loc, tag_0deg_loc, xyz_vr)
    test_vr = b2vr(ap_loc, tag_0deg_loc, test_board)
    # print(test_board)
    print('Test VR -> Board -> VR')
    print(test_vr, end='\t should ==\t')  # NOTE: test_vr should be same as xyz_vr
    print(xyz_vr, end='\n\n')

    xyz_b = (0, 0, 1)
    test_vr = b2vr(ap_loc, tag_0deg_loc, xyz_b)
    test_board = vr2b(ap_loc, tag_0deg_loc, test_vr)
    print('Test Board -> VR -> Board')
    print(test_board, end='\t should ==\t')  # NOTE: test_vr should be same as xyz_vr
    print(xyz_b)