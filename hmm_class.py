import preprocess as pp
import pomegranate as pg
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess import preprocess_any_file
class HiddenMarkovModel(object):

    def __init__(self, data):
        self.data = data


    def preprocess_data(data):
        train, test = train_test_split(data, test_size=0.0, shuffle=False)
        data_columns = list(data.columns.values)
        hidden_sequence = data['hypnogram_User'].tolist()
        for i in range(0, len(hidden_sequence)):
            if hidden_sequence[i] == "NotScored":
                hidden_sequence[i] = hidden_sequence[i - 1]

        observation_sequence = train.iloc[:, 0:n_features].values.tolist()
        state_names = np.unique(hidden_sequence).tolist()
        return data_columns, hidden_sequence, observation_sequence, state_names, train, test

    def create_distributions():
        wake_set = train[train.hypnogram_User == 'Wake']
        nonrem1_set = train[train.hypnogram_User == 'NonREM1']
        nonrem2_set = train[train.hypnogram_User == 'NonREM2']
        nonrem3_set = train[train.hypnogram_User == 'NonREM3']
        rem_set = train[train.hypnogram_User == 'REM']
        sets = [nonrem1_set, nonrem2_set, nonrem3_set, rem_set, wake_set]
        state_multidistributions = []
        for set in sets:
            state_dist = []
            for i in range(0, n_features):
                state_dist.append(pg.NormalDistribution.from_samples(set[data_columns[i]].tolist()))
            state_multidistributions.append(state_dist)

        return state_multidistributions

data, n_features = pp.preprocess_any_file('/Users/kristina/PycharmProjects/vyskumak/Data/8.3.2017-Z-F-50let.csv',
                                          n_features=10)
hmm = HiddenMarkovModel(data)
hmm.preprocess_data()
