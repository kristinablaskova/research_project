from typing import Dict, Iterable

import pomegranate as pg
from sklearn.metrics import confusion_matrix
from preprocess_functions import *
from sklearn.model_selection import train_test_split
from preprocess import df


data = df
def preprocess_data(data):
    train, test = train_test_split(data, test_size=0.0, shuffle=False)
    data_columns = list(data.columns.values)
    hidden_sequence = data[data_columns[10]].tolist()
    observation_sequence = train.iloc[:,0:10].values.tolist()
    state_names = np.unique(hidden_sequence).tolist()
    return data_columns, hidden_sequence, observation_sequence, state_names, train, test

data_columns, hidden_sequence, observation_sequence, state_names, train, test = preprocess_data(data)


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
        for i in range(0, 10):
            state_dist.append(pg.NormalDistribution.from_samples(set[data_columns[i]].tolist()))
        state_multidistributions.append(state_dist)

    return state_multidistributions

state_multidistributions = create_distributions()

def create_transition_probs():
    chain_model = pg.MarkovChain.from_samples([hidden_sequence])
    return chain_model

chain_model = create_transition_probs()
#creates empty hidden markov model with name 'prediction'
model = pg.HiddenMarkovModel('prediction')

states = {}  # type: Dict[str, pg.State]
dist = [ pg.IndependentComponentsDistribution(x) for x in state_multidistributions ]
for name in state_names:
    states[name] = pg.State(dist[state_names.index(name)], name=name)

#adds the states to the model
model.add_states(list(states.values()))
#sets the starting probability for state 'Wake' to 1.0
model.add_transition(model.start, states['Wake'], 1.0)
#insert the emission probabilities, that we computed in summary
for prob in chain_model.distributions[1].parameters[0]:
    state1 = states[prob[0]]
    state2 = states[prob[1]]
    probability = prob[2]
    model.add_transition(state1, state2, probability)
model.bake()

test_observation_sequence = train.iloc[:, 0:10].values.tolist()
hmm_fit = model.fit([observation_sequence], labels=[hidden_sequence], algorithm='labeled')
hmm_pred = model.predict(test_observation_sequence)

conf_hmm = confusion_matrix(hidden_sequence, [state_names[id] for id in hmm_pred], state_names)
print(conf_hmm)
print(state_names)

state_ids = np.array([state_names.index(val) for val in hidden_sequence])
score = (np.array(hmm_pred) == state_ids).mean()
print(score)