import pandas as pd
import sklearn.model_selection as ms
import os
import data_preprocessing as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn import svm

#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv("experimenty/list_of_patients_with_attributes.csv")

#LEARN MODEL ON FIRST TRAINING SET PATIENT
score=[]

for i in range(0, len(list_of_patients['file_name'])):
    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_testing_patients = list_of_testing_patients.reset_index()
    list_of_training_patients = list_of_patients.drop([i], axis=0)
    list_of_training_patients = list_of_training_patients.reset_index()


# PREPROCESS TESTING FILE
    testing_patient_path = "Data/"  + str(list_of_testing_patients['file_name'][0])
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


# PREPROCESS TRAINING FILES

    X_train = np.ndarray(shape=(0,34))
    y_train = []
    for j in range(list_of_training_patients.shape[0]):
        path = "Data/"+ str(list_of_training_patients['file_name'][j])
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


    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(test_observation_sequence)
    score = np.append(score,(y_pred == test_hidden_sequence).mean())
    print(score)