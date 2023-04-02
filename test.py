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

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(feature)
pred = kmeans.predict(feature)

with open('test_1_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)