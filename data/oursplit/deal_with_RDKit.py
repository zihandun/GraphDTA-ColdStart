"""
Created by Zihan Dun
2025/11/17
"""
import pandas as pd
import glob
from rdkit import Chem
import os

csv_files = glob.glob("*.csv")

def canonicalize_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

for csv_file in csv_files:
    print(f"Processing {csv_file} ...")
    df = pd.read_csv(csv_file)

    if 'compound_iso_smiles' in df.columns:
        df['compound_iso_smiles'] = df['compound_iso_smiles'].apply(canonicalize_smiles)
        out_file = os.path.splitext(csv_file)[0] + "_canonical.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved canonical SMILES to {out_file}")
    else:
        print(f"Skipped {csv_file}, no 'compound_iso_smiles' column found.")
