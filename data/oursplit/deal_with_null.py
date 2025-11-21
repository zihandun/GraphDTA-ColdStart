"""
Created by Zihan Dun
2025/11/17
"""
import os
import pandas as pd

# 获取当前目录下所有 CSV 文件
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

for file in csv_files:
    print(f"Processing {file}...")
    
    df = pd.read_csv(file)
    
    if 'affinity' in df.columns:
        # 删除 affinity 列为空（NaN）的行
        df = df[df['affinity'].notna()]
        df.to_csv(file, index=False)
        print(f"Updated {file}.")
    else:
        print(f"No 'affinity' column in {file}, skipped.")