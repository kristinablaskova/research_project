import os
from hmm import run_hmm_on_files
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        path =str(directory)[2:-1]+"/"+str(filename)
        run_hmm_on_files(path)