import sklearn.model_selection as ms
import data_preprocessing as dp

path = '/Users/kristina/PycharmProjects/vyskumak/3.5.2017-Z-F-57let.csv'
n_features =10
data, n_features = dp.preprocess_any_file(path, n_features)


def preprocess_data(data):
    train, test = ms.train_test_split(data, test_size=0.0, shuffle=False)
    data_columns = list(data.columns.values)
    hidden_sequence = data['hypnogram_User'].tolist()
    for i in range(0, len(hidden_sequence)):
        if hidden_sequence[i] == "NotScored":
            hidden_sequence[i] = hidden_sequence[i - 1]

    observation_sequence = train.iloc[:5, 0:2].values.tolist()
    state_names = np.unique(hidden_sequence).tolist()
    return data_columns, hidden_sequence, observation_sequence, state_names, train, test


data_columns, hidden_sequence, observation_sequence, state_names, train, test = preprocess_data(data)
'''
model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution,
    n_components = 5,
    X = observation_sequence,
    labels=hidden_sequence,
    algorithm='labeled'
)
'''