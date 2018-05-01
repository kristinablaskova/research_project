from sklearn.decomposition import PCA
import pandas as pd
from hmm import data

pca = PCA(n_components=34)
pca.fit(data.iloc[:,0:34])
print(pca.components_)


