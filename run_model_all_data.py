import os
from hmm import run_hmm_on_files
import pandas as pd
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')

experimental_df1 = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/experimenty/score_EEGonly_noTrainTest.csv')
experimental_df1 = experimental_df1.drop('Unnamed: 0', axis=1)
experimental_df1 = experimental_df1.rename(index=str, columns={"skore": "skore_gauss"})
experimental_df = pd.DataFrame(columns=['gauss_kernel_binar'])

index=0
for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        path =str(directory)[2:-1]+"/"+str(filename)
        score = run_hmm_on_files(path, n_features=10)
        experimental_df = experimental_df.append({'gauss_kernel_binar': score}, ignore_index=True)

experimental_df1= experimental_df1.reset_index(drop=True)
experimental_df= experimental_df.reset_index(drop=True)
experimental_df1 = experimental_df1.join(experimental_df)
experimental_df1.to_csv('1score_EEGonly_noTrainTest.csv', sep = ',')
