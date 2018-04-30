from typing import Dict, Iterable

import pandas as pd
import numpy as np
import pomegranate as pg
from sklearn.metrics import confusion_matrix

path = '3.5.2017-Z-F-57let.csv'
def preprocess_data(path):
    data = pd.read_csv(path, sep = ";")
    data = data.replace(',', '.', regex=True)
    data.columns = [c.replace('.', '_') for c in data.columns]
    data_columns = list(data.columns.values)
    hidden_sequence = data[data_columns[34]].tolist()
    observation_sequence = data[data_columns[0]].tolist()
    state_names = np.unique(hidden_sequence).tolist()
    return data, data_columns, hidden_sequence, observation_sequence, state_names

data, data_columns, hidden_sequence, observation_sequence, state_names = preprocess_data(path)

def create_distributions():
    wake_set = data[data.hypnogram_User == 'Wake']
    nonrem1_set = data[data.hypnogram_User == 'NonREM1']
    nonrem2_set = data[data.hypnogram_User == 'NonREM2']
    nonrem3_set = data[data.hypnogram_User == 'NonREM3']
    rem_set = data[data.hypnogram_User == 'REM']
    wake_dist = pg.NormalDistribution.from_samples(wake_set[data_columns[0]].tolist())
    nonrem1_dist = pg.NormalDistribution.from_samples(nonrem1_set[data_columns[0]].tolist())
    nonrem2_dist = pg.NormalDistribution.from_samples(nonrem2_set[data_columns[0]].tolist())
    nonrem3_dist = pg.NormalDistribution.from_samples(nonrem3_set[data_columns[0]].tolist())
    rem_dist = pg.NormalDistribution.from_samples(rem_set[data_columns[0]].tolist())
    return wake_dist, nonrem1_dist, nonrem2_dist, nonrem3_dist, rem_dist

wake_dist, nonrem1_dist, nonrem2_dist, nonrem3_dist, rem_dist = create_distributions()

def create_transition_probs():
    chain_model = pg.MarkovChain.from_samples([hidden_sequence])
    return chain_model

chain_model = create_transition_probs()
#creates empty hidden markov model with name 'prediction'
model = pg.HiddenMarkovModel('prediction')

states = {}  # type: Dict[str, pg.State]
dist = [wake_dist, nonrem1_dist, nonrem2_dist, nonrem3_dist, rem_dist]
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

hmm_fit = model.fit([observation_sequence], labels=[hidden_sequence], algorithm='labeled')
hmm_pred = model.predict(observation_sequence)

conf_hmm = confusion_matrix(hidden_sequence, [state_names[id] for id in hmm_pred], state_names)
print(conf_hmm)
print(state_names)

state_ids = np.array([state_names.index(val) for val in hidden_sequence])
score = (np.array(hmm_pred) == state_ids).mean()
print(score)