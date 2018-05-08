from preprocess_functions import *
import pandas as pd

#uses the functions from preprocess_functions to preprocess the data
def preprocess_any_file(path, n_features):
    prep = Preprocess()
    data = prep.data_import(path)
    X_feature, y, predictors = prep.prep_data_feature_selection(data)
    X_transformed, results, selector = prep.select_kbest(X_feature, y, n_features)
    features = prep.get_names(selector, X_feature)
    df = pd.DataFrame(X_transformed, columns = features)
    df['hypnogram_User'] = y
    return df, n_features, features