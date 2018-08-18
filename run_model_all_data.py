import os
from hmm import run_hmm_on_files
import pandas as pd
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
import preprocess as pr

#run hidden markov on all files separately
def run_separate(directory):
    experimental_df = pd.DataFrame(columns=['file', 'vysledok', 'poznamka'])
    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            path =str(directory)[2:-1]+"/"+str(filename)
            poznamka, score = run_hmm_on_files(path, n_features=10)
            experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},
                                                     ignore_index=True)

    experimental_df= experimental_df.reset_index(drop=True)
    experimental_df.to_csv('experimenty/zeny.csv', sep = ',')

#run_separate(directory)

#find top 10 features for the group of patients
def find_group_features(directory):
    features_all_patients = []
    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            path =str(directory)[2:-1]+"/"+str(filename)
            df, n_features, feature_names = pr.preprocess_any_file(path, n_features=10)
            features_all_patients.extend(feature_names)
    return features_all_patients


features_all_patients = find_group_features(directory)
df = pd.DataFrame()
df['features'] = pd.Series(features_all_patients).values
features_group_experiment = df['features'].value_counts(normalize=True)[:10].index.tolist()


#run hmm on all data as group data
def run_group(group_experiment_features, directory):
    cols = group_experiment_features.extend(['hypnogram_User'])
    data_array = []
    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            path =str(directory)[2:-1]+"/"+str(filename)
            prep = pr.Preprocess()
            data = prep.data_import(path)
            y = data['hypnogram_User'].copy().tolist()
            data_array = df.as_matrix()
    return cols, data
#cols, data = run_group(features_group_experiment, directory)

directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
list_of_patients_with_attributes = pd.DataFrame(['file_name','sex','age'])
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        start = filename.find('-Z-') + 3
        end = filename.find('let', start)
        sex_and_age = filename[start:end]
        sex = sex_and_age[0]
        age = sex_and_age[2:]
        list_of_patients_with_attributes = list_of_patients_with_attributes.append({'file_name': filename,
                                                                                    'sex': sex, 'age': age},
                                                                                   ignore_index=True)

list_of_patients_with_attributes = list_of_patients_with_attributes.dropna(subset=['age'])
list_of_patients_with_attributes = list_of_patients_with_attributes[list_of_patients_with_attributes['age']
                                                                    != '1.2016-AN-M-72']
list_of_patients_with_attributes['age'] = list_of_patients_with_attributes['age'].astype('int')
list_of_patients_with_attributes = list_of_patients_with_attributes.drop(list_of_patients_with_attributes.columns[0], axis=1)

females = list_of_patients_with_attributes[list_of_patients_with_attributes['sex'] == 'F']

experimental_df = pd.DataFrame(columns=['file', 'vysledok', 'poznamka'])
for filename in females['file_name']:

    if filename.endswith(".csv"):
        path = str(directory)[2:-1] + "/" + str(filename)
        print(path)
        poznamka, score = run_hmm_on_files(path, n_features=10)
        experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},
                                                 ignore_index=True)

experimental_df = experimental_df.reset_index(drop=True)
experimental_df.to_csv('experimenty/zeny.csv', sep=',')
