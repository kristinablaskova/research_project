import os
from hmm import run_hmm_on_files
import pandas as pd
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data_hradec')

experimental_df = pd.DataFrame(columns=['file', 'vysledok', 'poznamka'])
for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        path =str(directory)[2:-1]+"/"+str(filename)
        poznamka, score = run_hmm_on_files(path, n_features=10)
        experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka}, ignore_index=True)

experimental_df= experimental_df.reset_index(drop=True)
experimental_df.to_csv('experimenty/olomouc_experiment_gausskernel.csv', sep = ',')
