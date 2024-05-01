import freud
import numpy as np
import gsd.hoomd
import rowan
import coxeter


def prepare_data(
    traj_fn, 
    chosen_frame,
    num_neighbors,
    class_label,
):
    """
    Returns a M*N data matrix from a given trajectory.
    Rows:
    M is the number of (# of selected frames) * (# of particles).
    Columns:
    0:N-1 is the dimension of feature vectors. 
    The last column is the corresponding target.
    """
    data = []
    traj = gsd.hoomd.open(traj_fn)
    for f_n in chosen_frame:
        frame = traj[f_n]
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        nlist = system_temp.query(system[1], {"num_neighbors": num_neighbors, "exclude_ii": True}).toNeighborList()
        fix_nn_distances = nlist.distances
        fix_nn_segments = nlist.segments
        fix_nn_segments = np.hstack((fix_nn_segments, np.array([len(fix_nn_distances)])))
        fix_nlist_to_idx = nlist.point_indices

        for i in range(frame.particles.N):
            neighbors_idx = fix_nlist_to_idx[i * num_neighbors:(i+1) * num_neighbors]
            
            # determine the order of input features
            fix_nn_absrij = np.copy(fix_nn_distances[fix_nn_segments[i]:fix_nn_segments[i+1]])
            fix_nn_absrij /= np.max(fix_nn_absrij)
            aescending_order = np.argsort(fix_nn_absrij)
            
            # dij
            fix_nn_absrij = fix_nn_absrij[aescending_order]  
            
            # qij
            orien_i_to_ref = frame.particles.orientation[i]
            orien_j_to_ref = frame.particles.orientation[neighbors_idx]
            orien_ij = rowan.divide(orien_i_to_ref, orien_j_to_ref)
            orien_ij = orien_ij[aescending_order, :]
            orien_ij = orien_ij.flatten()

            
            #theta, phi_ij
            r_ij = system_temp.box.wrap(frame.particles.position[i] - frame.particles.position[neighbors_idx])
            r_ij /= np.linalg.norm(r_ij, axis=1).reshape(r_ij.shape[0],1)
            bond_angle_ij = appendSpherical_np(r_ij)[:, -2:]
            bond_angle_ij = bond_angle_ij[aescending_order, :]
            bond_angle_ij = bond_angle_ij.flatten()

            
            feature = np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij))
            data.append(np.hstack((feature, np.array([class_label]))))
    
    data = np.vstack(data)
    return data

def prepare_triangles_data(
    traj_fn, 
    chosen_frame,
    num_neighbors,
    class_label,
):
    """
    Returns a M*N data matrix from a given trajectory.
    Rows:
    M is the number of mini-batch ((end_frame - start_frame)*N_particles).
    Columns:
    0:N-1 is the dimension of feature vectors. 
    The last column is the corresponding target.
    """
    data = []
    traj = gsd.hoomd.open(traj_fn)
    for f_n in chosen_frame:
        frame = traj[f_n]
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        nlist = system_temp.query(system[1], {"num_neighbors": num_neighbors, "exclude_ii": True}).toNeighborList()
        fix_nn_distances = nlist.distances
        fix_nn_segments = nlist.segments
        fix_nn_segments = np.hstack((fix_nn_segments, np.array([len(fix_nn_distances)])))
        fix_nlist_to_idx = nlist.point_indices

        for i in range(frame.particles.N):
            # label particles based on typeid
            if frame.particles.typeid[i] == 2:
                label = 1
            else:
                label = class_label
            neighbors_idx = fix_nlist_to_idx[i * num_neighbors:(i+1) * num_neighbors]
            
            # determine the order of input features
            fix_nn_absrij = np.copy(fix_nn_distances[fix_nn_segments[i]:fix_nn_segments[i+1]])
            fix_nn_absrij /= np.max(fix_nn_absrij)
            aescending_order = np.argsort(fix_nn_absrij)
            
            # dij
            fix_nn_absrij = fix_nn_absrij[aescending_order]  
            
            # qij
            orien_i_to_ref = frame.particles.orientation[i]
            orien_j_to_ref = frame.particles.orientation[neighbors_idx]
            orien_ij = rowan.divide(orien_i_to_ref, orien_j_to_ref)
            orien_ij = orien_ij[aescending_order, :]
            orien_ij = orien_ij.flatten()

            
            #theta, phi_ij
            r_ij = system_temp.box.wrap(frame.particles.position[i] - frame.particles.position[neighbors_idx])
            r_ij /= np.linalg.norm(r_ij, axis=1).reshape(r_ij.shape[0],1)
            bond_angle_ij = appendSpherical_np(r_ij)[:, -2:]
            bond_angle_ij = bond_angle_ij[aescending_order, :]
            bond_angle_ij = bond_angle_ij.flatten()

            
            feature = np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij))
            data.append(np.hstack((feature, np.array([label]))))
    
    data = np.vstack(data)
    return data


def prepare_data_augmented(
    traj_fn, 
    chosen_frame,
    num_neighbors,
    class_label,
    D_theta=1,
    D_phi=1,
    ref_u=np.array([[0., 0., 1.]]),
    mirror=False
):
    """
    Returns a M*N data matrix from a given trajectory.
    Rows:
    M is the number of (# of selected frames) * (# of particles).
    Columns:
    0:N-1 is the dimension of feature vectors. 
    The last column is the corresponding target.
    """
    data = []
    traj = gsd.hoomd.open(traj_fn)
    for f_n in chosen_frame:
        frame = traj[f_n]
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        nlist = system_temp.query(system[1], {"num_neighbors": num_neighbors, "exclude_ii": True}).toNeighborList()
        fix_nn_distances = nlist.distances
        fix_nn_segments = nlist.segments
        fix_nn_segments = np.hstack((fix_nn_segments, np.array([len(fix_nn_distances)])))
        fix_nlist_to_idx = nlist.point_indices

        for i in range(frame.particles.N):
            neighbors_idx = fix_nlist_to_idx[i * num_neighbors:(i+1) * num_neighbors]
            
            # determine the order of input features
            fix_nn_absrij = np.copy(fix_nn_distances[fix_nn_segments[i]:fix_nn_segments[i+1]])
            fix_nn_absrij /= np.max(fix_nn_absrij)
            aescending_order = np.argsort(fix_nn_absrij)
            
            # dij
            fix_nn_absrij = fix_nn_absrij[aescending_order]
            
            
            # qij
            orien_i_to_ref = rowan.divide([1, 0, 0, 0], frame.particles.orientation[i])
            orien_j_to_ref = rowan.divide([1, 0, 0, 0], frame.particles.orientation[neighbors_idx])
            orien_ij = rowan.divide(orien_i_to_ref, orien_j_to_ref)
            orien_ij = orien_ij[aescending_order, :]
            orien_ij = normalize_qij(num_neighbors, orien_ij, D_theta, D_phi, ref_u=ref_u, mirror=mirror)
            orien_ij = orien_ij.flatten()

            
            #theta, phi_ij
            r_ij = system_temp.box.wrap(frame.particles.position[i] - frame.particles.position[neighbors_idx])
            r_ij /= np.linalg.norm(r_ij, axis=1).reshape(r_ij.shape[0],1)
            r_ij_to_ref = rowan.rotate(orien_i_to_ref, r_ij)
            bond_angle_ij = normalize_angleij(r_ij_to_ref, D_theta, D_phi, mirror=mirror)
            bond_angle_ij = bond_angle_ij[aescending_order, :]
            bond_angle_ij = bond_angle_ij.flatten()

            
            feature = np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij))
            data.append(np.hstack((feature, np.array([class_label]))))
    
    data = np.vstack(data)
    return data


def prepare_triangles_data_augmented(
    traj_fn, 
    chosen_frame,
    num_neighbors,
    class_label,
    D_theta=1,
    D_phi=1,
    ref_u=np.array([[0., 0., 1.]]),
    mirror=False
):
    """
    Returns a M*N data matrix from a given trajectory.
    Rows:
    M is the number of (# of selected frames) * (# of particles).
    Columns:
    0:N-1 is the dimension of feature vectors. 
    The last column is the corresponding target.
    """
    data = []
    traj = gsd.hoomd.open(traj_fn)
    for f_n in chosen_frame:
        frame = traj[f_n]
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        nlist = system_temp.query(system[1], {"num_neighbors": num_neighbors, "exclude_ii": True}).toNeighborList()
        fix_nn_distances = nlist.distances
        fix_nn_segments = nlist.segments
        fix_nn_segments = np.hstack((fix_nn_segments, np.array([len(fix_nn_distances)])))
        fix_nlist_to_idx = nlist.point_indices

        for i in range(frame.particles.N):
            # label particles based on typeid
            if frame.particles.typeid[i] == 2:
                label = 1
            else:
                label = class_label
            neighbors_idx = fix_nlist_to_idx[i * num_neighbors:(i+1) * num_neighbors]
            
            # determine the order of input features
            fix_nn_absrij = np.copy(fix_nn_distances[fix_nn_segments[i]:fix_nn_segments[i+1]])
            fix_nn_absrij /= np.max(fix_nn_absrij)
            aescending_order = np.argsort(fix_nn_absrij)
            
            # dij
            fix_nn_absrij = fix_nn_absrij[aescending_order]
            
            
            # qij
            orien_i_to_ref = rowan.divide([1, 0, 0, 0], frame.particles.orientation[i])
            orien_j_to_ref = rowan.divide([1, 0, 0, 0], frame.particles.orientation[neighbors_idx])
            orien_ij = rowan.divide(orien_i_to_ref, orien_j_to_ref)
            orien_ij = orien_ij[aescending_order, :]
            orien_ij = normalize_qij(num_neighbors, orien_ij, D_theta, D_phi, ref_u=ref_u, mirror=mirror)
            orien_ij = orien_ij.flatten()

            
            #theta, phi_ij
            r_ij = system_temp.box.wrap(frame.particles.position[i] - frame.particles.position[neighbors_idx])
            r_ij /= np.linalg.norm(r_ij, axis=1).reshape(r_ij.shape[0],1)
            r_ij_to_ref = rowan.rotate(orien_i_to_ref, r_ij)
            bond_angle_ij = normalize_angleij(r_ij_to_ref, D_theta, D_phi, mirror=mirror)
            bond_angle_ij = bond_angle_ij[aescending_order, :]
            bond_angle_ij = bond_angle_ij.flatten()

            
            feature = np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij))
            data.append(np.hstack((feature, np.array([class_label]))))
    
    data = np.vstack(data)
    return data


def normalize_angleij(r_ij, D_theta, D_phi, mirror=False):
    if mirror:
        # deal with mirror symmetry
        r_ij[:, 2] = np.abs(r_ij[:, 2])
    bond_angle_ij = appendSpherical_np(r_ij)[:, -2:]
    bond_angle_ij /= np.pi
    bond_angle_ij[:, 0] = np.remainder(bond_angle_ij[:, 0], 1/D_theta)
    bond_angle_ij[:, 1] = np.remainder(bond_angle_ij[:, 1], 2/D_phi)
    return bond_angle_ij


def normalize_qij(N_neighbors, qij, D_theta, D_phi, ref_u=np.array([[0., 0., 1.]]), mirror=False):
    spin_axis = rowan.rotate(qij, ref_u)
    spin_axis = spin_axis.reshape(N_neighbors, 3)
    if mirror:
        # deal with mirror symmetry
        spin_axis[:, 2] = np.abs(spin_axis[:, 2])
    s_dot_ref_u = np.einsum('ij, ij->i', spin_axis, ref_u)  # dot product, check if qij involves spin
    mask_if_rotat = np.abs(s_dot_ref_u) < 0.999999
    mask_if_antiparallel = s_dot_ref_u < -0.999999

    rotat_axis = np.repeat(ref_u, N_neighbors, axis=0)
    rotat_axis[mask_if_rotat, :] = np.cross(spin_axis[mask_if_rotat, :], ref_u)
    rotat_axis[mask_if_antiparallel, :] = np.roll(ref_u, shift=1)
    rotat_theta_norm = np.repeat(0.0, N_neighbors)
    rotat_theta_norm[mask_if_rotat] = np.remainder(np.arccos(s_dot_ref_u[mask_if_rotat]), np.pi / D_theta)
    if D_theta != 1:
        rotat_theta_norm[mask_if_antiparallel] = np.remainder(np.pi, np.pi / D_theta)
    else:
        rotat_theta_norm[mask_if_antiparallel] = np.pi
    
    rotat_q_norm = rowan.from_axis_angle(rotat_axis, rotat_theta_norm)
    
    spin_q = rowan.divide(qij, rotat_q_norm)
    ## solve unstable numerical results from rowan.divide
    check_pos_cos_val = spin_q[:, 0] >= 1.0
    check_neg_cos_val = spin_q[:, 0] <= -1.0
    spin_q[check_pos_cos_val, :] = np.array([1.0, 0.0, 0.0, 0.0])
    spin_q[check_neg_cos_val, :] = np.array([-1.0, 0.0, 0.0, 0.0])
    
    spin_axis_from_q, spin_phi = rowan.to_axis_angle(spin_q)
    mask_if_parallel = np.einsum('ij, ij->i', spin_axis, spin_axis_from_q) < 0.0  # dot product, check if the spin phi return by rowan follows the convention
    spin_phi_norm = spin_phi
    spin_phi_norm[mask_if_parallel] = 2*np.pi - spin_phi_norm[mask_if_parallel]
    spin_phi_norm = np.remainder(spin_phi_norm, 2 * np.pi / D_phi)
    
    spin_q_norm = rowan.from_axis_angle(spin_axis, spin_phi_norm)

    q_norm = rowan.multiply(spin_q_norm, rotat_q_norm)  # By convention, apply rotation then spin
    return q_norm

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
