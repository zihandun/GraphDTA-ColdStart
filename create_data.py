import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
from utils import *
from collections import OrderedDict


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                               'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                               'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]

    return c_size, features, edge_index


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x


compound_iso_smiles = set()
for dt_name in ['kiba', 'davis']:
    opts = ['train', 'test', 'val', 'test1', 'test2']
    for opt in opts:
        df = pd.read_csv(f'data/oursplit/{dt_name}_{opt}_canonical.csv')
        compound_iso_smiles.update(df['compound_iso_smiles'])

smile_graph = {smile: smile_to_graph(smile) for smile in compound_iso_smiles}

datasets = ['davis', 'kiba']
splits = ['train', 'test', 'val', 'test1', 'test2']

for dataset in datasets:
    for split in splits:
        processed_data_file = f'data/processed/{dataset}_{split}.pt'
        if not os.path.isfile(processed_data_file):
            csv_file = f'data/oursplit/{dataset}_{split}_canonical.csv'
            df = pd.read_csv(csv_file)

            drugs = np.asarray([s for s in df['compound_iso_smiles']])
            prots = np.asarray([seq_cat(s) for s in df['target_sequence']])
            Y = np.asarray([y for y in df['affinity']])

            print(f'Preparing {dataset}_{split}.pt in PyTorch format...')
            _ = TestbedDataset(root='data', dataset=f'{dataset}_{split}', xd=drugs, xt=prots, y=Y,
                               smile_graph=smile_graph)
            print(f'{processed_data_file} has been created')
        else:
            print(f'{processed_data_file} is already created')
