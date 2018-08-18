import os
import pandas as pd
import hmm as myhmm


directory = os.fsencode('./Data')
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
        poznamka, score = myhmm.run_hmm_on_files(path, n_features=10)
        experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},
                                                 ignore_index=True)

experimental_df = experimental_df.reset_index(drop=True)
experimental_df.to_csv('experimenty/zeny.csv', sep=',')


# hypnogram of average person
# prep = Preprocess()
# average_person = prep.data_import('/Users/kristina/PycharmProjects/vyskumak/Data/12.10.2016-Z-M-39let.csv')
# average_person['time'] = [(0.5/60*i) for i in range(0, 973)]
# cleanup_nums = {"hypnogram_User": {"Wake": 5, "REM": 4, "NonREM1": 3, "NonREM2": 2, "NonREM3": 1}}
# average_person.replace(cleanup_nums, inplace=True)
# plt.plot(average_person['time'], average_person['hypnogram_User'])
# y = [5, 4, 3, 2, 1]
# labels = ["Wake", "REM", "NonREM1", "NonREM2", "NonREM3"]
# plt.yticks(y, labels)
# plt.xlabel('Measurement length [hours]')
# plt.ylabel('Sleep stage')
# plt.title('Hypnogram of average person from dataset Hradec Kralove')
# plt.show()


# age histogram
# plt.hist(list_of_patients_with_attributes['age'],10, color='grey')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.title('Histogram of age from Hradec Kralove dataset')
# plt.axvline(list_of_patients_with_attributes['age'].mean(), color='orange', linestyle='dashed', linewidth=1)
# plt.axvline(list_of_patients_with_attributes['age'].median(), color='yellow', linestyle='dashed', linewidth=1)
# plt.show()
#
# sex histogram
# plt.xlabel('Sex')
# plt.ylabel('Frequency')
# plt.title('Histogram of sex from Hradec Kralove dataset')
# plt.xticks(rotation=90)
# list_of_patients_with_attributes['sex'].value_counts().plot(kind = 'bar', color='grey')


# path =str(directory)[2:-1] + "/" + str(filename)
# poznamka, score = run_hmm_on_files(path, n_features=10)
# experimental_df = experimental_df.append({'file': filename, 'vysledok': score, 'poznamka': poznamka},ignore_index=True)
# experimental_df = experimental_df.reset_index(drop=True)
# experimental_df.to_csv('experimenty/new.csv', sep = ',')