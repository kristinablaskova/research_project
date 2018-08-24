from typing import Dict, Iterable
import pandas as pd
import os
import pomegranate as pg
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import data_preprocessing as dp
import numpy as np
import dist as dst

#run hmm separately on files
def run_hmm_on_files(path, n_features):
    poznamka = []
    try:
        print(path)

        data, n_features, feature_names = dp.preprocess_any_file(path, n_features)


        def preprocess_data(data):
            train, test = ms.train_test_split(data, test_size=0.3, shuffle=False)
            data_columns = list(data.columns.values)
            hidden_sequence = data['hypnogram_User'].tolist()
            l = len(hidden_sequence)
            for i in reversed(range(0, l)):
                if hidden_sequence[i] == "NotScored":
                    train = train.drop([i])
                    del hidden_sequence[i]
            observation_sequence = train.iloc[:, 0:n_features].values.tolist()
            return data_columns, hidden_sequence, observation_sequence, train, test

        def create_states(model, hidden_sequence, state_names):
            chain_model = pg.MarkovChain.from_samples([hidden_sequence])
            states = {}  # type: Dict[str, pg.State]
            for name in state_names:
                states[name] = pg.State(dist[state_names.index(name)], name=name)
            model.add_states(list(states.values()))
            # sets the starting probability for state 'Wake' to 1.0
            try:
                model.add_transition(model.start, states['Wake'], 1.0)
                poznamka.append("")
            except KeyError:
                print("nezacina wake")
                poznamka.append('nezacina wake')
                pass
            # insert the emission probabilities, that we computed in summary
            for prob in chain_model.distributions[1].parameters[0]:
                state1 = states[prob[0]]
                state2 = states[prob[1]]
                probability = prob[2]
                model.add_transition(state1, state2, probability)



        data_columns, hidden_sequence, observation_sequence, train, test = preprocess_data(data)
        hmm_dist = dst.Distributions(train)
        dist, state_names = hmm_dist.gauss_kernel_dist(feature_names)
        model = pg.HiddenMarkovModel('prediction')
        create_states(model, hidden_sequence, state_names)
        model.bake()

        test_observation_sequence = train.iloc[:, 0:n_features].values.tolist()
        #hmm_fit = model.fit([observation_sequence], labels=[hidden_sequence], algorithm='labeled')
        hmm_pred = model.predict(test_observation_sequence)

        conf_hmm = metrics.confusion_matrix(hidden_sequence, [state_names[id] for id in hmm_pred], state_names)
        #print(conf_hmm)
        #print(state_names)

        state_ids = np.array([state_names.index(val) for val in hidden_sequence])
        score = (np.array(hmm_pred) == state_ids).mean()
        print(score)
    except ValueError:
        print('nejaky valueerror - napr nepozna stlpec hypnogram user')
        score = "NaN"
        feature_names = []

    return poznamka, score

#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv("/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv")
list_of_training_patients, list_of_testing_patients = ms.train_test_split(list_of_patients, test_size=0.3, shuffle=True)

#LEARN MODEL ON FIRST TRAINING SET PATIENT
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')

list_of_training_patients = list_of_training_patients.reset_index()
path = str(directory)[2:-1] + "/" + str(list_of_training_patients['file_name'][0])
poznamka, score = run_hmm_on_files(path, n_features=10)
#run hmm on files as group
'''
def run_hmm_on_files(path, n_features):
    try:
        print(path)

        data, n_features, feature_names = preprocess_any_file(path, n_features)


        def preprocess_data(data):
            train, test = train_test_split(data, test_size=0.0, shuffle=False)
            data_columns = list(data.columns.values)
            hidden_sequence = data['hypnogram_User'].tolist()
            l = len(hidden_sequence)
            for i in reversed(range(0, l)):
                if hidden_sequence[i] == "NotScored":
                    train = train.drop([i])
                    del hidden_sequence[i]
            observation_sequence = train.iloc[:, 0:n_features].values.tolist()
            return data_columns, hidden_sequence, observation_sequence, train, test

        def create_states(model, hidden_sequence, state_names):
            chain_model = pg.MarkovChain.from_samples([hidden_sequence])
            states = {}  # type: Dict[str, pg.State]
            for name in state_names:
                states[name] = pg.State(dist[state_names.index(name)], name=name)
            model.add_states(list(states.values()))
            # sets the starting probability for state 'Wake' to 1.0
            try:
                model.add_transition(model.start, states['Wake'], 1.0)
            except KeyError:
                print("nezacina wake")
                pass
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
    except ValueError:
        print('nejaky valueerror - napr nepozna stlpec hypnogram user')
        score = "NaN"
        feature_names = []

    return poznamka, score
'''