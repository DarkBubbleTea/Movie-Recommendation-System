import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pickle
import warnings
warnings.filterwarnings('ignore')

uni = pd.read_csv('cwurData.csv')
uni.drop('broad_impact', axis=1, inplace=True)
uni_des = uni.describe()
uni.insert(0, 'index', uni.index)

features = uni[['quality_of_education', 'quality_of_faculty', 'influence', 'score']]

features_np = features.to_numpy()
sum_dist = []
K = range(1,15)
for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(features_np)
    sum_dist.append(k_means.inertia_)

# Run after run for loop above
# plt.plot(K, sum_dist, 'bx-')
# plt.show()
# Plot show best n_clusters is 4

k_mean_4 = KMeans(n_clusters=4)
model = k_mean_4.fit(features_np)
result = k_mean_4.labels_

features_1 = uni[['index', 'quality_of_education', 'quality_of_faculty', 'influence', 'score']]
features_2 = uni[['index', 'world_rank', 'institution', 'country', 'alumni_employment', 'citations', 'patents']]
lookup = features_1.merge(features_2, on='index', how='left')
lookup['cluster'] = result

def uni_recommend(model, qualityeducation, qualityfaculty, influence, score):
    arr = np.array([[qualityeducation, qualityfaculty, influence, score]])
    pred = model.predict(arr)
    return lookup[lookup['cluster'] == pred[0]].sample(5)

a = uni_recommend(model, 150, 80, 80, 80)