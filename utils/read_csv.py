import STRING
import pandas as pd

def reading_files_transporte(name_file):
    file = pd.read_csv(name_file, delimiter=';', encoding='latin1')
    return file