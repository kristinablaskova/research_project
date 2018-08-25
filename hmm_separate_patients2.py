import pandas as pd
import sklearn.model_selection as ms
import os
import data_preprocessing as dp
import dist as dst
import pomegranate as pg
import numpy as np
import sklearn.metrics as metrics


#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv("experimenty/list_of_patients_with_attributes.csv")
#LEARN MODEL ON FIRST TRAINING SET PATIENT
poznamka = []

def preprocess_data(data):
    train, test = ms.train_test_split(data, test_size=0.0, shuffle=False)
    data_columns = list(data.columns.values)
    hidden_sequence = train['hypnogram_User'].tolist()
    l = len(hidden_sequence)
    for i in reversed(range(0, l)):
        if hidden_sequence[i] == "NotScored":
            train = train.drop([i])
            del hidden_sequence[i]
    train = train.drop(['hypnogram_Machine'], axis=1)
    test = test.drop(['hypnogram_Machine'], axis=1)

    observation_sequence = train.iloc[:, 0:n_features].values.tolist()
    return data_columns, hidden_sequence, observation_sequence, train, test

def create_states(model, hidden_sequence, state_names):
    chain_model = pg.MarkovChain.from_samples(hidden_sequence)
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

for i in range(0, len(list_of_patients['file_name'])):
    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_training_patients = list_of_patients.drop([i], axis=0)

    list_of_training_patients = list_of_training_patients.reset_index()
    training_feature_array = []
    training_class_array = []
    train_df = pd.DataFrame()

    for i in range(list_of_training_patients.shape[0]):
        path = "Data/"+ str(list_of_training_patients['file_name'][0])
        patient_data = dp.data_import(path)
        binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                                   "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
        for feature in binary_features:
            if feature in patient_data.columns:
                patient_data = patient_data.drop(feature, axis=1)
        df1 = patient_data.pop('hypnogram_User')
        patient_data['hypnogram_User'] = df1
        n_features = patient_data.shape[1] - 2
        data_columns, hidden_sequence, observation_sequence, train, test = preprocess_data(data=patient_data)
        training_class_array.append(hidden_sequence)
        train_df = train_df.append(train)

    hmm_dist = dst.Distributions(train_df)
    feature_names = patient_data.drop(['hypnogram_User', 'hypnogram_Machine'], axis=1).columns.values.tolist()
    dist, state_names = hmm_dist.gauss_kernel_dist(feature_names)

    model = pg.HiddenMarkovModel('prediction')
    create_states(model, training_class_array, state_names)
    model.bake()

    #TESTING PART!!! :)
    list_of_testing_patients = list_of_testing_patients.reset_index()

    for i in range(list_of_testing_patients.shape[0]):
        path = "Data/"+ str(list_of_testing_patients['file_name'][i])
        patient_data = dp.data_import(path)
        binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                                   "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
        for feature in binary_features:
            if feature in patient_data.columns:
                patient_data = patient_data.drop(feature, axis=1)
        df1 = patient_data.pop('hypnogram_User')
        patient_data['hypnogram_User'] = df1
        n_features = patient_data.shape[1] - 2
        data_columns, hidden_sequence, observation_sequence, train1, test = preprocess_data(data=patient_data)

        test_observation_sequence = train1.iloc[:, 0:n_features].values.tolist()

        # REWRITTEN model.predict()
        viterbi_result = model.viterbi(test_observation_sequence)
        if viterbi_result[1] is None:
            print('Dropping patient')
            continue

        hmm_pred = [state_id for state_id, state in viterbi_result[1]]
        ##########

        # SANITY CHECKS
        for v in range(0, len(hmm_pred)):
            if 4 < hmm_pred[v] or hmm_pred[v] < 0:
                print(v, hmm_pred[v])

        print(len(hidden_sequence), len(hmm_pred))
        ##########

        conf_hmm = metrics.confusion_matrix(hidden_sequence, [state_names[min(id, 4)] for id in hmm_pred][1:], state_names)
        print(conf_hmm)
        print(state_names)
        state_ids = np.array([state_names.index(val) for val in hidden_sequence])
        score = (np.array(hmm_pred[1:]) == state_ids).mean()
        print(score)
