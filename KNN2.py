import pandas as pd
import sklearn.model_selection as ms
import os
import data_preprocessing as dp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv("/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv")
#LEARN MODEL ON FIRST TRAINING SET PATIENT
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')

list_of_testing_patients = list_of_patients[:10]
list_of_testing_patients = list_of_testing_patients.reset_index()
list_of_training_patients = list_of_patients[10:]
list_of_training_patients = list_of_training_patients.reset_index()

# PREPROCESS TRAINING FILES

X_train = np.ndarray(shape=(0,34))
y_train = []
for i in range(list_of_training_patients.shape[0]):
    path = str(directory)[2:-1] + "/" + str(list_of_training_patients['file_name'][0])
    patient_data = dp.data_import(path)
    binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                               "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
    for feature in binary_features:
        if feature in patient_data.columns:
            patient_data = patient_data.drop(feature, axis=1)
    df1 = patient_data.pop('hypnogram_User')
    patient_data['hypnogram_User'] = df1
    patient_data = patient_data.drop(['hypnogram_Machine'], axis=1)
    train_observation_sequence = patient_data.iloc[:, :-1].values
    train_hidden_sequence = patient_data.iloc[:, -1].values
    X_train = np.append(train_observation_sequence, X_train, axis=0)
    y_train = np.append(train_hidden_sequence, y_train)

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)

# PREPROCESS TESTING FILE
for j in range(list_of_testing_patients.shape[0]):
    testing_patient_path = str(directory)[2:-1] + "/" + str(list_of_testing_patients['file_name'][j])
    testing_patient_data = dp.data_import(testing_patient_path)
    binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                       "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
    for feature in binary_features:
        if feature in testing_patient_data.columns:
            testing_patient_data = testing_patient_data.drop(feature, axis=1)
    df1 = testing_patient_data.pop('hypnogram_User')
    testing_patient_data['hypnogram_User'] = df1
    testing_patient_data = testing_patient_data.drop(['hypnogram_Machine'], axis=1)

    test_observation_sequence = testing_patient_data.iloc[:, :-1].values
    test_hidden_sequence = testing_patient_data.iloc[:, -1].values

    y_pred = classifier.predict(test_observation_sequence)

    print(confusion_matrix(test_hidden_sequence, y_pred))
    print(classification_report(test_hidden_sequence, y_pred))