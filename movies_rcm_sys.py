# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
df = pd.read_csv('movies.csv')
drop_col = ['homepage', 'keywords', 'tagline', 'crew',
            'revenue', 'production_countries', 'budget', 'production_companies']
df.drop(drop_col, axis=1, inplace=True)
df['release_date'] = df['release_date'].str[:4]
df.dropna(inplace=True)
spoken_lan_pattern = r'"iso_639_1": "(\w+)"'
df['spoken_languages'] = df['spoken_languages'].str.extractall(spoken_lan_pattern).groupby(level=0).apply(lambda x: ','.join(x[0]))
bins = [0, 30, 90, 120, float('inf')]
labels = ['Very Short', 'Short', 'Medium', 'Long']
df['runtime_cat'] = pd.cut(df['runtime'], bins=bins, labels=labels)
df.insert(9, 'runtime_cat', df.pop('runtime_cat'))
cutoffs = [0, 4.668070, 12.921594, 28.313505, float('inf')]
popu_labels = ['Very Low', 'Low', 'Medium', 'High']
df['popularity_cat'] = pd.cut(df['popularity'], bins=cutoffs, labels=popu_labels)
df.insert(7, 'popularity_cat', df.pop('popularity_cat'))
ori_lan_cat = []
for i in df['original_language']:
    if i == 'en':
        ori_lan_cat.append(i)
    else:
        ori_lan_cat.append('Others')

se = pd.Series(ori_lan_cat)
df['original_language_cat'] = se.values

# Features
feature = df[['id', 'original_language_cat', 'popularity_cat', 'runtime_cat', 'status', 'vote_average']]
feature['popularity_cat'].fillna(feature['popularity_cat'].mode()[0], inplace=True)
feature['runtime_cat'].fillna(feature['runtime_cat'].mode()[0], inplace=True)

# Encode category features
le = LabelEncoder()

for i in feature.columns[1:-1]:
    feature[i] = le.fit_transform(feature[i])

# Build model
feature_np = feature.to_numpy()
sum_dist = []
K = range(1,15)
for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(feature_np)
    sum_dist.append(k_means.inertia_)

# Run after run for loop above
# plt.plot(K, sum_dist, 'bx-')
# plt.show()
# Plot show best n_clusters is 5

k_mean_5 = KMeans(n_clusters=5)
model = k_mean_5.fit(feature_np)
result = k_mean_5.labels_
print(silhouette_score(feature_np, result, metric='euclidean'))
print(calinski_harabasz_score(feature_np, result))

# Create 2 new dataframe then merge them. Use it to show result
feature_1 = df[['id', 'original_language_cat', 'popularity_cat', 'runtime_cat', 'status', 'vote_average']]
feature_2 = df[['id', 'genres', 'original_language', 'original_title', 'popularity',
                'runtime', 'spoken_languages', 'title', 'vote_count']]
lookup = feature_1.merge(feature_2, on='id', how='left')
lookup['cluster'] = result


# lookup.to_csv('lookup.csv')

# def movies_recommend(model, language, popular, runtime, status, vote):
#     arr = []
#     for col, val in zip(['original_language_cat', 'popularity_cat', 'runtime_cat', 'status', 'vote_average'], [language, popular, runtime, status, vote]):
#         if val in le.classes_:
#             arr.append(le.transform([val])[0])
#         else:
#             arr.append(-1)
#     arr.append(vote)
#     arr = np.array(arr).reshape(1, -1)
#     pred = model.predict(arr)
#
#     cluster = pred[0]
#     mask = lookup['cluster'] == cluster
#     return lookup[mask].sample(5)
#
# a = movies_recommend(model, 'en', 'Medium', 'Long', 'Release', 8)

with open('test_1_model.pkl', 'wb') as f:
    pickle.dump((model, le), f)