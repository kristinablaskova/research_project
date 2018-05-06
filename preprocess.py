from preprocess_functions import *
import pandas as pd

#uses the functions from preprocess_functions to preprocess the data
def preprocess_any_file(path):
    data = data_import(path)
    X_feature, y, predictors = prep_data_feature_selection(data)
    X_transformed, results = select_kbest(X_feature, y, 10)
    best_features_sorted = get_names(predictors, results)
    df = pd.DataFrame(X_transformed, columns = best_features_sorted[:10])
    df['hypnogram_User'] = y
    return df

