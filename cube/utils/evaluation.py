from .data_prep import appendSpherical_np, normalize_angleij, normalize_qij

import torch
import torch.nn as nn
import torch.nn.functional as F

import freud
import numpy as np
import gsd.hoomd
import rowan


def mlp_eval(num_neighbors, traj_fn, chosen_frame, model, n_class):
    dnn_ops = []
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

        dnn_op = np.zeros(n_class)
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

            feature = torch.tensor(np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij)), dtype=torch.float)
            with torch.no_grad():
                output = model(feature.reshape(1, len(feature)))
                pred = output.data.max(1, keepdim=True)[1]
                dnn_op[pred.data] += 1

        dnn_ops.append(dnn_op / frame.particles.N)
    return np.vstack(dnn_ops)


def mlp_eval_augmented(num_neighbors, traj_fn, chosen_frame, model, n_class,
                       D_theta=1,
                       D_phi=1,
                       ref_u=np.array([[0., 0., 1.]]),
                       mirror=False
                       ):
    dnn_ops = []
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

        dnn_op = np.zeros(n_class)
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


            feature = torch.tensor(np.hstack((fix_nn_absrij, orien_ij, bond_angle_ij)), dtype=torch.float)
            with torch.no_grad():
                output = model(feature.reshape(1, len(feature)))
                pred = output.data.max(1, keepdim=True)[1]
                dnn_op[pred.data] += 1

        dnn_ops.append(dnn_op / frame.particles.N)
    return np.vstack(dnn_ops)

def Q_class(num_neighbors, traj_fn, chosen_frame, l, th=0.55):
    q = []
    traj = gsd.hoomd.open(traj_fn)
    for f_n in chosen_frame:
        frame = traj[f_n]
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        nlist = system_temp.query(system[1], {"num_neighbors": num_neighbors, "exclude_ii": True}).toNeighborList()
        ql = freud.order.Steinhardt(l=l, average=True)
        ql.compute(system, neighbors=nlist)
        qi_sc = ql.particle_order > th
        
        class_fraction = np.sum(qi_sc) / qi_sc.shape[0]
        q.append(class_fraction)
    return q
    
def calc_rdf(r_max, traj_fn, start_f, end_f, steps, bins=30):
    g_r = []
    bin_edges = []
    rdf = freud.density.RDF(bins, r_max)
    
    traj = gsd.hoomd.open(traj_fn)
    for frame in traj[start_f:end_f:steps]:
        system = (frame.configuration.box, frame.particles.position)
        system_temp = freud.AABBQuery(*system)
        rdf.compute(system=system_temp)
        bin_edges.append(rdf.bin_edges)
        g_r.append(rdf.rdf)
        
    
    return np.vstack(bin_edges), np.vstack(g_r)