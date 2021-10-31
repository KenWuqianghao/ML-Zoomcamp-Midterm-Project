import pandas as pd
import numpy as np
import pickle

columns = ['cnt', 'timestamp', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']

df = pd.read_csv('data.csv', usecols=columns)

df['timestamp'] = df['timestamp'].str.slice(10,13)
df['timestamp'] = df['timestamp'].astype(int).astype(str)

from sklearn.model_selection import train_test_split

df['cnt']=np.log1p(df['cnt'])

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.cnt.values
y_val = df_val.cnt.values
y_test = df_test.cnt.values

del df_train['cnt']
del df_val['cnt']
del df_test['cnt']

from sklearn.feature_extraction import DictVectorizer

train_dict = df_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(val_dict)

X_val = dv.transform(val_dict)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, max_depth = 100, random_state = 1, n_jobs = -1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

with open('model.bin', 'wb') as f_out:
   pickle.dump((dv, rf), f_out)
f_out.close()