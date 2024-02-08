import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch
from utils.rotation_transforms import axis_angle_to_matrix


def apply_torsion(data, omega, homogeneous=True):

    if not homogeneous:
        # different molecules
        data.pos = data.pos + 0  # shallow copy
        graphs = data.to_data_list()
        cum_idx_edge = 0
        cum_idx_node = 0

        for G in graphs:
            N, M, R = G.num_nodes, G.num_edges, G.edge_mask.sum()

            pos, edge_index, edge_mask, mask_rotate = G.pos + 0, G.edge_index, G.edge_mask, G.mask_rotate[0][0]
            omega_G = omega[cum_idx_edge: cum_idx_edge + R]
            cum_idx_edge += R

            edge_index = edge_index.T[edge_mask]
            for idx_edge, e in enumerate(edge_index):
                u, v = e[0], e[1]

                # check if need to reverse the edge, v should be connected to the part that gets rotated
                assert not mask_rotate[idx_edge, u]
                assert mask_rotate[idx_edge, v]

                rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
                rot_mat = axis_angle_to_matrix(rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * omega_G[idx_edge:idx_edge + 1])

                pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v:v + 1]) @ torch.transpose(rot_mat, 0, 1) + pos[v:v + 1]

            data.pos[cum_idx_node: cum_idx_node + N] = pos
            cum_idx_node += N


    else:
        # batched for graphs of the same molecule
        B = data.num_graphs
        N, M, R = data.num_nodes // B, data.num_edges // B, data.edge_mask.sum() // B

        pos, edge_index, edge_mask, mask_rotate = data.pos.reshape(B, N, 3) + 0, data.edge_index[:, :M], data.edge_mask[:M], data.mask_rotate[0][0][0]
        omega = omega.reshape(B, R)

        edge_index = edge_index.T[edge_mask]
        for idx_edge, e in enumerate(edge_index):
            u, v = e[0], e[1]

            # check if need to reverse the edge, v should be connected to the part that gets rotated
            assert not mask_rotate[idx_edge, u]
            assert mask_rotate[idx_edge, v]

            rot_vec = pos[:, u] - pos[:, v]  # convention: positive rotation if pointing inwards
            rot_mat = axis_angle_to_matrix(rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * omega[:, idx_edge:idx_edge+1])

            pos[:, mask_rotate[idx_edge]] = torch.bmm(pos[:, mask_rotate[idx_edge]] - pos[:, v:v+1], torch.transpose(rot_mat, 1, 2)) + pos[:, v:v+1]

        data.pos = pos.reshape(-1, 3)




def dx_dtau(pos, edge, mask, batch=True):
    if batch:
        uu, v = pos[:,edge].unbind(1)
        bond = uu - v
        bond = bond / torch.linalg.norm(bond, dim=-1, keepdims=True)
        u_side, v_side = pos[:,~mask] - uu[:,None], pos[:,mask] - uu[:,None]
        u_side, v_side = torch.linalg.cross(u_side, bond[:,None], dim=-1), torch.cross(v_side, bond[:,None], dim=-1)
    else:
        uu, v = pos[edge].unbind(0)
        bond = uu - v
        bond = bond / torch.linalg.norm(bond, dim=-1, keepdims=True)
        u_side, v_side = pos[~mask] - uu[None], pos[mask] - uu[None]
        u_side, v_side = torch.linalg.cross(u_side, bond[None], dim=-1), torch.cross(v_side, bond[None], dim=-1)
    return u_side, v_side


def log_det_jac(data, homogeneous=True):
    
    if not homogeneous:
        graphs = data.to_data_list()
        result = torch.zeros(len(graphs), device=data.pos.device)

        for i, G in enumerate(graphs):
            N, M, R = G.num_nodes, G.num_edges, G.edge_mask.sum()

            pos, edge_index, edge_mask, mask_rotate = G.pos + 0, G.edge_index, G.edge_mask, G.mask_rotate[0][0]
            edge_index = edge_index.T[edge_mask]

            jac = []
            for edge, mask in zip(edge_index, mask_rotate):
                dx_u, dx_v = dx_dtau(pos, edge, mask, batch=False)
                dx = torch.zeros_like(pos)
                dx[~mask] = dx_u
                jac.append(dx)

            jac_rot = torch.linalg.cross(pos[:, None], torch.eye(3).to(pos)[None])  # N, 3, 3, first 3 is basis vectors
            jac_rot = jac_rot.unbind(-2)
            jac = torch.stack(jac + list(jac_rot), 0)  # R+3, N, 3
            jac = (jac - jac.mean(-2, keepdims=True)).reshape(R + 3, 3 * N)
            result[i] = torch.log(torch.linalg.svdvals(jac)).sum(-1)

    else:
        B = data.num_graphs
        N, M, R = data.num_nodes // B, data.num_edges // B, data.edge_mask.sum() // B
        pos, edge_index, edge_mask, mask_rotate = data.pos.reshape(B, N, 3) + 0, data.edge_index[:, :M], data.edge_mask[:M], data.mask_rotate[0][0][0]
        edge_index = edge_index.T[edge_mask]

        jac = []
        for edge, mask in zip(edge_index, mask_rotate):
            dx_u, dx_v = dx_dtau(pos, edge, mask, batch=True)
            dx = torch.zeros_like(pos)
            dx[:,~mask] = dx_u
            jac.append(dx)

        jac_rot = torch.linalg.cross(pos[:,:,None], torch.eye(3).to(pos)[None,None]) # B, N, 3, 3, first 3 is basis vectors
        jac_rot = jac_rot.unbind(-2)
        jac = torch.stack(jac + list(jac_rot), 1) # B, R+3, N, 3
        jac = (jac - jac.mean(-2, keepdims=True)).reshape(B, R+3, 3*N)
        result = torch.log(torch.linalg.svdvals(jac)).sum(-1)
    return result
    


def get_transformation_mask(pyg_data, only_single_bonds=True):

    G = to_networkx(pyg_data, to_undirected=False)
    if not nx.is_connected(G.to_undirected()):
        raise Exception("Ligand graph is not connected, are there multiple ligands?")
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]


        
        if only_single_bonds and pyg_data.edge_attr0[i, 0] != 0:
            to_rotate.append([])
            to_rotate.append([])
            continue

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    # TODO deal with case of edge being the centroid (connected components of the same size)
    return mask_edges, mask_rotate

