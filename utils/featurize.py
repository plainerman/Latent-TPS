import numpy as np
import torch, io
from torch_geometric.data import HeteroData, Data
from openmm import app
from rdkit.Chem import rdmolfiles

from utils.general import pdbfile_writeFooter
from utils.torsion import get_transformation_mask

atom_features_list = {
    'atomic_num': list(range(1, 75)) + ['misc'],  # stop at tungsten
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'numring': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'numring': [0, 1, 2, 'misc'],
    'implicit_valence': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'numH': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'numH': [0, 1, 2, 3, 4, 'misc'],
    'number_radical_e': [0, 1, 2, 3, 4, 'misc'],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'is_aromatic': [False, True],
    'is_in_ring3': [False, True],
    'is_in_ring4': [False, True],
    'is_in_ring5': [False, True],
    'is_in_ring6': [False, True],
    'is_in_ring7': [False, True],
    'is_in_ring8': [False, True],
    'residues': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                 'A', 'C', 'G', 'I', 'U', 'DA', 'DC', 'DG', 'DI', 'DU', 'N',
                 'SEC', 'PYL', 'ASX', 'GLX', 'UNK',
                 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                 'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'atom_types': ['C', 'C*', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1',
                   'CG2', 'CH', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'N*', 'ND', 'ND1', 'ND2', 'NE', 'NE1',
                   'NE2', 'NH', 'NH1', 'NH2', 'NZ', 'O', 'O*', 'OD', 'OD1', 'OD2', 'OE', 'OE1', 'OE2',
                   'OG', 'OG1', 'OH', 'OX', 'OXT', 'S*', 'SD', 'SG', 'misc']
}
bond_features_list = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'],
    'bond_stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
    'is_conjugated': [False, True]
}


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


def featurize_atoms(args, mol):
    atom_features = []
    ringinfo = mol.GetRingInfo()

    def safe_index_(key, val):
        return safe_index(atom_features_list[key], val)

    for idx, atom in enumerate(mol.GetAtoms()):
        features = [
            safe_index_('atomic_num', atom.GetAtomicNum()),
            safe_index_('chirality', str(atom.GetChiralTag())),
            safe_index_('degree', atom.GetTotalDegree()),
            safe_index_('numring', ringinfo.NumAtomRings(idx)),
            safe_index_('implicit_valence', atom.GetImplicitValence()),
            safe_index_('formal_charge', atom.GetFormalCharge()),
            safe_index_('numH', atom.GetTotalNumHs()),
            safe_index_('hybridization', str(atom.GetHybridization())),
            safe_index_('is_aromatic', atom.GetIsAromatic()),
            safe_index_('is_in_ring5', ringinfo.IsAtomInRingOfSize(idx, 5)),
            safe_index_('is_in_ring6', ringinfo.IsAtomInRingOfSize(idx, 6)),
            safe_index_('residues', atom.GetPDBResidueInfo().GetResidueName().strip()),
            safe_index_('atom_types', atom.GetPDBResidueInfo().GetName().strip())
        ]

        if args.node_embeddings:
            features.append(idx)
        atom_features.append(features)

    return torch.tensor(atom_features)


def get_feature_dims(args, dataset):
    node_feature_dims = [
        len(atom_features_list['atomic_num']),
        len(atom_features_list['chirality']),
        len(atom_features_list['degree']),
        len(atom_features_list['numring']),
        len(atom_features_list['implicit_valence']),
        len(atom_features_list['formal_charge']),
        len(atom_features_list['numH']),
        len(atom_features_list['hybridization']),
        len(atom_features_list['is_aromatic']),
        len(atom_features_list['is_in_ring5']),
        len(atom_features_list['is_in_ring6']),
        len(atom_features_list['residues']),
        len(atom_features_list['atom_types'])
    ]

    if args.node_embeddings:
        N = dataset.target.topology.getNumAtoms()
        node_feature_dims.append(N)

    edge_feature_dims = [
        len(bond_features_list['bond_type']),
        len(bond_features_list['bond_stereo']),
        len(bond_features_list['is_conjugated'])
    ]
    if args.edge_embeddings:
        N = dataset.target.topology.getNumAtoms()
        edge_feature_dims.append(int(N * (N - 1) + 1))

    return node_feature_dims, edge_feature_dims


def featurize_bond(bond):
    bond_feature = [
        safe_index(bond_features_list['bond_type'], str(bond.GetBondType())),
        safe_index(bond_features_list['bond_stereo'], str(bond.GetStereo())),
        safe_index(bond_features_list['is_conjugated'], bond.GetIsConjugated())
    ]
    return bond_feature


def get_graph(args, topology, hetero=False):
    stringio = io.StringIO()
    close, stringio.close = stringio.close, lambda: None
    app.PDBFile.writeModel(topology, np.zeros((topology.getNumAtoms(), 3)), stringio)
    pdbfile_writeFooter(topology, stringio, allBonds=True)
    block = stringio.getvalue()
    stringio.close = close;
    stringio.close()
    mol = rdmolfiles.MolFromPDBBlock(block, removeHs=False, proximityBonding=False)

    data = Data() if not hetero else HeteroData()

    atom_feats = featurize_atoms(args, mol)
    data.node_attr0 = atom_feats

    bond_idx, bond_attr = get_bond_edges(mol)

    if args.edges == 'dense':
        dense_idx = get_dense_edges(atom_feats.shape[0])
        if args.edge_embeddings:
            dense_attr = torch.arange(dense_idx.shape[1])[:, None] + 1
        else:
            dense_attr = torch.zeros(dense_idx.shape[1], 0)
        if args.hetero:
            edge_index, edge_attr0 = combine_edges((bond_idx, bond_attr), (dense_idx, dense_attr))
            data['root', 'root'].edge_index = data['prop', 'prop'].edge_index = edge_index
            data['root', 'root'].edge_attr0 = data['prop', 'prop'].edge_attr0 = edge_attr0
        else:
            data.edge_index, data.edge_attr0 = combine_edges((bond_idx, bond_attr), (dense_idx, dense_attr))

    elif args.edges == 'radius':
        if hetero:
            data['root', 'root'].edge_index = data['prop', 'prop'].edge_index = bond_idx
            data['root', 'root'].edge_attr0 = data['prop', 'prop'].edge_attr0 = bond_attr
        else:
            data.edge_index, data.edge_attr0 = bond_idx, bond_attr
    else:
        NotImplemented

    edge_mask, mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(edge_mask)
    data.mask_rotate = [[torch.tensor(mask_rotate)]]
    return data


def combine_edges(*edges):
    idxs, feats = zip(*edges)
    idx = torch.cat(idxs, -1)
    feat_dim = sum(f.shape[-1] for f in feats)
    feat = torch.zeros((idx.shape[-1], feat_dim), dtype=torch.float32)
    r, c = 0, 0
    for f in feats:
        rr, cc = f.shape
        feat[r:r + rr, c:c + cc] = f
        r += rr;
        c += cc;
    return idx, feat


def get_bond_edges(mol):
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat += [featurize_bond(bond)]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_feat = torch.tensor(edge_feat)
    edge_feat = torch.concat([edge_feat, edge_feat], 0)

    return edge_index, edge_feat.type(torch.uint8)


def get_dense_edges(n):
    atom_ids = np.arange(n)
    src, dst = np.repeat(atom_ids, n), np.tile(atom_ids, n)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_idx = np.stack([src, dst])
    return torch.tensor(edge_idx)
