import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import itertools
import numpy as np
import numpy.linalg as la
from algo import axis


def loc_weighted(aoa_data_synced, weight, setups):
    # weight: n_pak x n_ap
    aps = sorted(list(aoa_data_synced.keys()))
    n_aps = len(aps)
    assert n_aps > 1, 'Must have at least 2 APs!'
    n_points = len(aoa_data_synced[aps[0]]['t_data'])
    locations = np.zeros([n_points, 3])
    r = 2

    PA = np.zeros([len(aps), 3], dtype=np.float)
    for ap, nth_ap in zip(aps, range(len(aps))):
        PA[nth_ap] = setups['ap_loc'][ap]

    if weight is None:
        weight = np.ones([n_points, n_aps], dtype=np.float)
    for i in range(n_points):  # For each packet
        PB = np.zeros([len(aps), 3], dtype=np.float)
        for ap, nth_ap in zip(aps, range(len(aps))):
            # Step 4
            theta = np.radians(aoa_data_synced[ap]['theta'][i])
            phi = np.radians(aoa_data_synced[ap]['phi'][i])
            x = - r * np.sin(phi)
            y = r * np.cos(phi) * np.sin(theta)
            z = - r * np.cos(phi) * np.cos(theta)
            # Step 5
            PB[nth_ap] = axis.b2vr(ap_loc=setups['ap_loc'][ap], tag_0_deg_loc=setups['tag_0deg_loc'], xyz_b=[x, y, z])
        locations[i], _ = lineIntersect3D_weighted(PA, PB, weight[i].T)

    return locations

# def loc_weighted_localmax(device_1_data, device_2_data, device_3_data, device_4_data, weight, verbose=False):
#   AP1_LOC = device_1_data['ap']
#   AP2_LOC = device_2_data['ap']
#   AP3_LOC = device_3_data['ap']
#   AP4_LOC = device_4_data['ap']
#   TAG_ZERO_LOC = device_1_data['0deg']
#   PA = np.vstack([AP1_LOC, AP2_LOC, AP3_LOC, AP4_LOC])
#   PO = np.array(TAG_ZERO_LOC)
#   facing_v_all = PA - PO
#   y_v = np.array([0, 1, 0])
#   x_v = np.array([1, 0, 0])
#   alpha = [np.sign((facing_v) @ x_v.T) * np.arccos((facing_v @ y_v.T) / (la.norm(facing_v) * la.norm(y_v))) for facing_v
#            in facing_v_all]
#
#   # Note: b2vr means from Board axis to VR axis
#   rotation_axis_b2vr = np.array([
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0]
#   ])
#
#   def rotation_orientation_b2vr(angle):
#     return np.array([
#       [1, 0, 0],
#       [0, np.cos(angle), np.sin(angle)],
#       [0, -np.sin(angle), np.cos(angle)]
#     ])
#
#   rotation_matrix_b2vr = np.array([rotation_axis_b2vr @ rotation_orientation_b2vr(i) for i in alpha])
#
#   # Get AP Theta Phi and trim to same length (of packets)
#   ap1 = device_1_data['out'] * np.pi / 180
#   ap2 = device_2_data['out'] * np.pi / 180
#   ap3 = device_3_data['out'] * np.pi / 180
#   ap4 = device_4_data['out'] * np.pi / 180
#   ap_data_list = [ap1, ap2, ap3, ap4]
#   n_points = np.amin([i.shape[0] for i in ap_data_list])
#   ap_data = np.stack([i[:n_points] for i in ap_data_list])
#
#   # Solve for location
#   locations = np.zeros([n_points, 3])
#   selected_thetas = np.zeros([n_points, 4])
#   selected_phis = np.zeros([n_points, 4])
#   confidence = np.zeros(n_points)
#
#   r = 2
#   for i in range(n_points):  # For each packet
#     # Make combinations of each local maxima theta, phi
#     thetas_comb = []
#     phis_comb = []
#     for ap in range(4):
#       thetas_comb.append(list(ap_data[ap][i, 0]))
#       phis_comb.append(list(ap_data[ap][i, 1]))
#     thetas_comb = list(itertools.product(*thetas_comb))
#     phis_comb = list(itertools.product(*phis_comb))
#
#     locations_comb = np.zeros([len(thetas_comb), 3])
#     distances_comb = np.zeros([len(thetas_comb), 4])
#     all_thetas = np.zeros([len(thetas_comb), 4])
#     all_phis = np.zeros([len(thetas_comb), 4])
#
#     for thetas, phis, kth_combo in zip(thetas_comb, phis_comb, range(len(thetas_comb))):
#       PB = np.zeros([4, 3])
#       for nth_ap in range(4):
#         # Step 4
#         theta, phi = thetas[nth_ap], phis[nth_ap]
#         x = - r * np.sin(phi)
#         y = - r * np.cos(phi) * np.sin(theta)
#         z = r * np.cos(phi) * np.cos(theta)
#         # Step 5
#         PB[nth_ap] = rotation_matrix_b2vr[nth_ap] @ np.array([x, y, z]) + PA[nth_ap]
#         all_thetas[kth_combo, nth_ap] = np.rad2deg(theta)
#         all_phis[kth_combo, nth_ap] = np.rad2deg(phi)
#       locations_comb[kth_combo], distances_comb[kth_combo] = lineIntersect3D_weighted(PA, PB, weight[i].T)
#     residues = np.sum(distances_comb, axis=1)
#     min_comb = np.argmin(residues)
#     locations[i] = locations_comb[min_comb]
#     selected_thetas[i] = all_thetas[min_comb]
#     selected_phis[i] = all_phis[min_comb]
#     confidence[i] = residues[min_comb]
#
#   return locations, selected_thetas, selected_phis, confidence

def loc_single_anchor(ap_results_data, ap_loc, tag_0deg_loc):
    r = ap_results_data['r']
    thetas_data = ap_results_data['theta']
    phis_data = ap_results_data['phi']
    n_packets = r.shape[0]

    locations = np.zeros([n_packets, 3])
    for i in range(n_packets):  # For each packet
        theta = np.radians(thetas_data[i])
        phi = np.radians(phis_data[i])
        x = - r[i] * np.sin(phi)
        y = r[i] * np.cos(phi) * np.sin(theta)
        z = - r[i] * np.cos(phi) * np.cos(theta)
        location_board_axis = np.array([x, y, z])
        locations[i] = axis.b2vr(ap_loc, tag_0deg_loc, location_board_axis)

    return locations


def lineIntersect3D_weighted(PA, PB, w):
    """
    Find intersection point of lines in 3D space, in the least squares sense
    :param PA: Nx3-matrix containing starting point of N lines
    :param PB: Nx3-matrix containing end point of N lines
    :param w: Nx1-vector containing weights of N lines
    :return: Best intersection point of the N lines, in least squares sense
    """
    w = np.divide(w, np.sum(w))
    Si = np.subtract(PB, PA)
    ni = np.divide(Si, np.multiply(np.expand_dims(np.sqrt(np.sum(np.power(Si, 2), axis=1)), axis=1), np.ones((1, 3))))
    nx, ny, nz = ni[:, 0], ni[:, 1], ni[:, 2]
    SXX = np.sum(np.multiply(w, np.subtract(np.power(nx, 2), 1)))
    SYY = np.sum(np.multiply(w, np.subtract(np.power(ny, 2), 1)))
    SZZ = np.sum(np.multiply(w, np.subtract(np.power(nz, 2), 1)))
    SXY = np.sum(np.multiply(w, np.multiply(nx, ny)))
    SXZ = np.sum(np.multiply(w, np.multiply(nx, nz)))
    SYZ = np.sum(np.multiply(w, np.multiply(ny, nz)))
    S = [[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]]
    #     print(S)
    CX = np.sum(np.multiply(w, np.add(np.add(np.multiply(PA[:, 0], np.subtract(np.power(nx, 2), 1)),
                                             np.multiply(PA[:, 1], np.multiply(nx, ny))),
                                      np.multiply(PA[:, 2], np.multiply(nx, nz)))))
    CY = np.sum(np.multiply(w, np.add(np.add(np.multiply(PA[:, 0], np.multiply(nx, ny)),
                                             np.multiply(PA[:, 1], np.subtract(np.power(ny, 2), 1))),
                                      np.multiply(PA[:, 2], np.multiply(ny, nz)))))
    CZ = np.sum(np.multiply(w, np.add(np.add(np.multiply(PA[:, 0], np.multiply(nx, nz)),
                                             np.multiply(PA[:, 1], np.multiply(ny, nz))),
                                      np.multiply(PA[:, 2], np.subtract(np.power(nz, 2), 1)))))
    C = np.transpose(np.array([CX, CY, CZ]))
    P_intersect = np.transpose(np.matmul(np.linalg.inv(S), C))

    N = np.size(PA, 0)
    distances = np.zeros(N)
    for i in range(N):
        ui = (P_intersect - PA[i, :]) @ Si[i, :].T / (Si[i, :] @ Si[i, :].T)
        distances[i] = la.norm(P_intersect - PA[i, :] - ui * Si[i, :])
        distances[i] *= w[i]
    return P_intersect, distances

def apply_ratb(loc1, loc2):
    """
    Tranformation is applied to loc1
    loc1, loc2 = set of 3d points, np.ndarray[n_pts, 3]
    """

    U, _, V = np.linalg.svd((loc1 - np.median(loc1, axis=1)[:, None]).T @ \
                            (loc2 - np.median(loc2, axis=1)[:, None]), full_matrices=True)
    R = V.T @ U.T
    if np.linalg.det(R) < 0:
        U, _, V = np.linalg.svd(R)
        V = V.T
        V[:, 2] = -V[:, 2]
        R = V @ U.T

    t = np.mean(loc2 - (R @ loc1[..., None])[..., 0], axis=0)
    loc1_ratb = np.squeeze(R @ loc1[..., None] + t[:, None])
    return loc1_ratb