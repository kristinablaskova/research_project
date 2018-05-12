import os
from hmm import run_hmm_on_files
import pandas as pd
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')

def run_separate(directory):
    experimental_df = pd.DataFrame(columns=['file', 'vysledok', 'poznamka'])
    features_all_patients = []
    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            path =str(directory)[2:-1]+"/"+str(filename)
            poznamka, score, feature_names = run_hmm_on_files(path, n_features=10)
            experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},
                                                     ignore_index=True)
            features_all_patients.extend(feature_names)

    experimental_df= experimental_df.reset_index(drop=True)
    experimental_df.to_csv('experimenty/hradec_new_gausskernel.csv', sep = ',')
    return features_all_patients

features_all_patients = run_separate(directory)
df = pd.DataFrame()
df['features'] = pd.Series(features_all_patients).values
features_group_experiment = df['features'].value_counts(normalize=True)[:10].index.tolist()