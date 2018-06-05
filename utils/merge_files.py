import os
import pandas as pd
import tqdm

def merge_files(file_dir: str, merge_file_dir:str, name:str, output_name:str, folder_name=''):

    files = list(set([f for f in os.listdir(file_dir+'\\' + folder_name)]))
    files = [f for f in files if name in f]
    df = pd.read_csv(files[0], sep=';', encoding='latin1')
    del files[0]

    for i in files:
        print(i)
        df_i = pd.read_csv(i, sep=';', encoding='latin1')
        df = df.append(df_i, ignore_index=True)
        del df_i

    df.to_csv(output_name, index=False, encoding='latin1', sep=';')

