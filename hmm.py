from typing import Dict, Iterable

import pomegranate as pg
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess import *
from dist import Distributions


def run_hmm_on_files(path, n_features):
    print(path)

    data, n_features, feature_names = preprocess_any_file(path, n_features)


    def preprocess_data(data):
        train, test = train_test_split(data, test_size=0.0, shuffle=False)
        data_columns = list(data.columns.values)
        hidden_sequence = data['hypnogram_User'].tolist()
        for i in range(0, len(hidden_sequence)):
            if hidden_sequence[i] == "NotScored":
                hidden_sequence[i] = hidden_sequence[i - 1]

        observation_sequence = train.iloc[:, 0:n_features].values.tolist()
        return data_columns, hidden_sequence, observation_sequence, train, test

    def create_states(model, hidden_sequence, state_names):
        chain_model = pg.MarkovChain.from_samples([hidden_sequence])
        states = {}  # type: Dict[str, pg.State]
        for name in state_names:
            states[name] = pg.State(dist[state_names.index(name)], name=name)
        model.add_states(list(states.values()))
        # sets the starting probability for state 'Wake' to 1.0
        model.add_transition(model.start, states['Wake'], 1.0)
        # insert the emission probabilities, that we computed in summary
        for prob in chain_model.distributions[1].parameters[0]:
            state1 = states[prob[0]]
            state2 = states[prob[1]]
            probability = prob[2]
            model.add_transition(state1, state2, probability)



    data_columns, hidden_sequence, observation_sequence, train, test = preprocess_data(data)
    hmm_dist = Distributions(train)
    dist, state_names = hmm_dist.gauss_kernel_dist(feature_names)
    model = pg.HiddenMarkovModel('prediction')
    create_states(model, hidden_sequence, state_names)
    model.bake()

    test_observation_sequence = train.iloc[:, 0:n_features].values.tolist()
    #hmm_fit = model.fit([observation_sequence], labels=[hidden_sequence], algorithm='labeled')
    hmm_pred = model.predict(test_observation_sequence)

    conf_hmm = confusion_matrix(hidden_sequence, [state_names[id] for id in hmm_pred], state_names)
    #print(conf_hmm)
    #print(state_names)

    state_ids = np.array([state_names.index(val) for val in hidden_sequence])
    score = (np.array(hmm_pred) == state_ids).mean()
    print(score)
    return score
