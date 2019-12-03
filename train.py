import sklearn
#sklearn.show_versions()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

# load data
dataset_url = '/home/kacham/Documents/datasets_kaggle/discrete/london_merged.csv'
data = pd.read_csv(dataset_url)

# separate features
y_features = ['cnt']
y = data[y_features]
y = y.astype('int') #fix 1
X_features = ['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']
X = data[X_features]
X = X.astype('int') #fix 1

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123) #fix 2, no stratify

# train a Gaussian classifier
model = GaussianNB()
model.fit(X_train, y_train.values.ravel()) # fix 3

# predict Output
predicted_new_bike_shares = model.predict(X_test)

# stats
print('--- comparison of real value and pred ---')
print(y_test)
print(predicted_new_bike_shares)
print('-----------------------------------------')

acc = accuracy_score(y_test, predicted_new_bike_shares)
f1 = f1_score()
mae = mean_absolute_error(y_test, predicted_new_bike_shares)
mse = mean_squared_error(y_test, predicted_new_bike_shares)
pr = precision_score()
r2 = r2_score(y_test, predicted_new_bike_shares)
rec = recall_score()
metrics_msg = 'accuracy: {} - f1: {} - mean absolute errror: {} - mean squared error: {} \n - precision: {} - r2 score: {} - recall: {} '.format(acc, f1, mae, mse, pr, r2, rec)
# "/home/kacham/Documents/tracelogs/metrics/(exp)_(challenge)_(model)_buggy/corrected_metrics.txt"
with open("/home/kacham/Documents/tracelogs/metrics/sk_not_calling_array_london_merged_GaussianNB_buggy_metrics.txt", "a") as myfile:
    myfile.write(metrics_msg)

# TODO write stats in files, maybe look at other interesting stats, such as Emilio's ?
