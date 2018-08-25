import pandas as pd
import data_preprocessing as dp


list_of_patients_with_attributes = pd.read_csv(
    "experimenty/list_of_patients_with_attributes.csv")

testing_patient_path = "Data/"  + str(list_of_patients_with_attributes['file_name'][0])
testing_patient_data = dp.data_import(testing_patient_path)
testing_patient_series = testing_patient_data['EEG_F3_A2: DELTA'].values.tolist()
list_of_training_patients = list_of_patients_with_attributes.drop([0], axis=0)
list_of_training_patients = list_of_training_patients.reset_index()

for j in range(1, len(list_of_training_patients['file_name'])):
    path = "Data/" + str(list_of_training_patients['file_name'][j])
    print(path)
    #data = dp.data_import(path)



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