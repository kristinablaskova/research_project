import pandas as pd

path = "/Users/kristina/PycharmProjects/vyskumak/experimenty/1score_EEGonly_noTrainTest.csv"
df = pd.read_csv(path)
df['datum'], df['zdravie'], df['pohlavie'], df['vek'], df['nezname'] = df['pacient'].str.split('-').str
df['spinacopy'] = df['nezname']
for i in range(0,df.shape[0]):
    if df['zdravie'][i] == "kinect" or df['zdravie'][i] == 'kinekt':
        df['spinacopy'][i] = df['zdravie'][i]
        df['zdravie'][i] = df['pohlavie'][i]
        df['pohlavie'][i] = df['vek'][i]
        df['vek'][i] = df['nezname'][i]
        df['nezname'] = df['spinacopy']
df = df.drop(['spinacopy'], axis = 1)
df.to_csv('extracted_properties.csv', sep = ',')
