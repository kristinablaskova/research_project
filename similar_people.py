import os
import pandas as pd
import hmm as myhmm
import matplotlib.pyplot as plt
import data_preprocessing as dp


directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
list_of_patients_with_attributes = pd.read_csv(
    "/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv")

females = list_of_patients_with_attributes[list_of_patients_with_attributes['sex'] == 'F']
males = list_of_patients_with_attributes[list_of_patients_with_attributes['sex'] == 'M']

# for female_file in females['file_name'][:2]:
#     path = str(directory)[2:-1] + "/" + str(filename)
#     data = dp.data_import(path)
#     data['EEG_F3_A2: DELTA'] = pd.to_numeric(data['EEG_F3_A2: DELTA'])
#     plt.plot(data['EEG_F3_A2: DELTA'])
#     plt.show()



# #just hmm run
# experimental_df = pd.DataFrame(columns=['file', 'vysledok', 'poznamka'])
# for filename in females['file_name']:
#
#     if filename.endswith(".csv"):
#         path = str(directory)[2:-1] + "/" + str(filename)
#         print(path)
#         poznamka, score = myhmm.run_hmm_on_files(path, n_features=10)
#         experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},
#                                                  ignore_index=True)
#
# experimental_df = experimental_df.reset_index(drop=True)
# experimental_df.to_csv('experimenty/zeny.csv', sep=',')





# path =str(directory)[2:-1] + "/" + str(filename)
# poznamka, score = run_hmm_on_files(path, n_features=10)
# experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},ignore_index=True)
# experimental_df = experimental_df.reset_index(drop=True)
# experimental_df.to_csv('experimenty/new.csv', sep = ',')