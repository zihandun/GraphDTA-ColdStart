"""
Created by Zihan Dun
2025/11/17
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict

def get_removelist(list_name, length):
    removelist = []
    for i, x in enumerate(list_name):
        if len(x) >= length:
            removelist.append(i)
    return removelist

def list_remove(list_name, removelist):
    idx = set(range(len(list_name))) - set(removelist)
    return [list_name[i] for i in idx]

def df_remove(dataframe, removelist, axis):
    if axis == 0:
        new_df = dataframe.drop(removelist).reset_index(drop=True)
    else:
        new_df = dataframe.drop(removelist, axis=1)
        new_df.columns = range(new_df.shape[1])
    return new_df


datasets = ['kiba', 'davis']

for dataset in datasets:
    print('Processing dataset:', dataset)
    fpath = f'data/{dataset}/'

    with open(f'{fpath}/data_split_oogrid.pkl', 'rb') as f:
        trainset, validset, testset, testset1, testset2 = pickle.load(f)

    with open(fpath + "ligands_can.txt") as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(fpath + "proteins.txt") as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)

    drug_list = list(ligands.values())
    prot_list = list(proteins.values())

    if dataset == 'davis':
        affinities = pd.read_csv(
            fpath + 'drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt',
            sep='\s+', header=None, encoding='latin1'
        )
        affinities = -(np.log10(affinities / 1e9))
        affinity = affinities.to_numpy()

    else:
        affinities = pd.read_csv(
            fpath + 'kiba_binding_affinity_v2.txt',
            sep='\s+', header=None, encoding='latin1'
        )

        ligands_remove = get_removelist(drug_list, 90)
        proteins_remove = get_removelist(prot_list, 1365)

        drug_list = list_remove(drug_list, ligands_remove)
        prot_list = list_remove(prot_list, proteins_remove)

        affinities = df_remove(affinities, ligands_remove, 0)
        affinities = df_remove(affinities, proteins_remove, 1)

        affinity = affinities.to_numpy()

    def write_csv(filename, pair_list):
        with open(filename, 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for drug_idx, target_idx in pair_list:
                smiles = drug_list[drug_idx]
                seq = prot_list[target_idx]
                aff = affinity[drug_idx, target_idx]
                f.write(f"{smiles},{seq},{aff}\n")

    write_csv(f'data/{dataset}_train.csv', trainset)
    write_csv(f'data/{dataset}_val.csv', validset)
    write_csv(f'data/{dataset}_test.csv', testset)
    write_csv(f'data/{dataset}_test1.csv', testset1)
    write_csv(f'data/{dataset}_test2.csv', testset2)

    print(f"{dataset} done: train={len(trainset)}, val={len(validset)}, test={len(testset)}, test1={len(testset1)}, test2={len(testset2)}")
