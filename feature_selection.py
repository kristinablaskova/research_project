import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
import numpy as np
import preprocess as pr
import os
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data_hradec')

#imports data and replace , with . to read numbers as float
class Preprocess(object):

    def __init__(self):
        pass

    def data_import(self, path):
        data = pd.read_csv(path, sep = ";")
        data = data.replace(',', '.', regex=True)
        data.columns = [c.replace('.', '_') for c in data.columns]
        data = data.loc[:, (data != 0).any(axis=0)]
        return data

    #prepares the data for feature selection
    def prep_data_feature_selection(self, data):
        X_feature = data.drop(['hypnogram_Machine', 'hypnogram_User'], axis=1).copy()
        predictors = X_feature.columns.values.tolist()
        y = data['hypnogram_User']
        return X_feature, y, predictors

    #KBest function - helps us select the relevant features
    def select_kbest(self, X_feature, y, number_of_besties):
        selector = SelectKBest(k=number_of_besties, score_func=f_classif)
        selector.fit(X_feature, y)
        results = -np.log10(selector.pvalues_)
        X_transformed = selector.fit_transform(X_feature, y).copy()
        return X_transformed, results, selector

    #Percentile function - helps us select the relevant features
    def select_percentile(self, X_feature, y, percentile):
        selector= SelectPercentile(percentile=percentile, score_func=f_classif)
        selector.fit(X_feature, y)
        results = -np.log10(selector.pvalues_)
        X_transformed = selector.fit_transform(X_feature, y).copy()
        return X_transformed, results

    #see results of percentile or kbest function
    def get_names(self, selector, X_feature):
        feature_names = []
        for i in range(0, X_feature.shape[1]):
            if selector.get_support()[i]:
                feature_names.append(X_feature.columns[i])
        return feature_names

    #find top 10 features for the group of patients
    def find_group_features(self, directory):
        features_all_patients = []
        for file in os.listdir(directory):

            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                path = str(directory)[2:-1] + "/" + str(filename)
                df, n_features, feature_names = pr.preprocess_any_file(path, n_features=10)
                features_all_patients.extend(feature_names)
        return features_all_patients

