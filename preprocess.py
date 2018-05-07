from preprocess_functions import *
import pandas as pd

#uses the functions from preprocess_functions to preprocess the data
def preprocess_any_file(path):
    data = data_import(path)
    X_feature, y, predictors = prep_data_feature_selection(data)
    X_transformed, results, selector = select_kbest(X_feature, y, 10)
    features = get_names(selector, X_feature)
    df = pd.DataFrame(X_transformed, columns = features)
    df['hypnogram_User'] = y
    return df