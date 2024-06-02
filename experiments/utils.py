"""Utility functions for experiments."""
import os
import numpy as np
import torch
from typing import Optional
import random
import torch.distributed as dist
from openfold.utils import rigid_utils

Rigid = rigid_utils.Rigid


import copy
def reorder_structure(input_feat, ind_res):
    #reorder_feats['torsion_angles_sin_cos'].shape  torch.Size([1, 160, 7, 2])
    res = copy.deepcopy(input_feat)
    for i in range(len(ind_res)):
        res['torsion_angles_sin_cos'][0, i] = input_feat['torsion_angles_sin_cos'][0, ind_res[i]]
        res['sc_ca_t'][:,i] = input_feat['sc_ca_t'][:,ind_res[i]]
        res['rigids_t'][:,i,:] = input_feat['rigids_t'][:,ind_res[i],:]
        res['motif_seq_masks'][:,:,i] = input_feat['motif_seq_masks'][:,:,ind_res[i]]
        res['motif_position'][:,i] = input_feat['motif_position'][:,ind_res[i]]
    return res

def tsp(distance_matrix, start):
    n = len(distance_matrix)
    visited = [start]
    path = [start]
    while len(visited) < n:
        min_distance = float('inf')
        next_node = None
        for i in range(n):
            if i not in visited:
                for j in path:
                    if distance_matrix[i][j] < min_distance:
                        min_distance = distance_matrix[i][j]
                        next_node = i
        visited.append(next_node)
        path.append(next_node)
    return path

from scipy.spatial import distance_matrix
def reorder(coords:np.array, motif_position):
    # N, CA, C, O
    num_motif, seq_len = motif_position.shape
    dist_matrix = distance_matrix(coords[:,0,:], coords[:,2,:])

    for i in range(num_motif):
        for j in range(1, seq_len):
            if motif_position[i][j-1] and motif_position[i][j]:
                dist_matrix[j, :] = 1e9
                dist_matrix[:, j-1] = 1e9
                dist_matrix[j, j-1] = 0.01

    min_values = np.min(dist_matrix, axis=1)
    start = np.argmax(min_values)

    concat_map = tsp(dist_matrix, start)
    return concat_map

def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_sampled_mask(contigs, length, rng=None, num_tries=1000000):
    '''
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.
    '''
    length_compatible=False
    count = 0
    while length_compatible is False:
        inpaint_chains=0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        #allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f'{contig_list[-1]},0'
        for con in contig_list:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0'):
                #receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            else:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
        #check length is compatible 
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count+=1
        if count == num_tries: #contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains
