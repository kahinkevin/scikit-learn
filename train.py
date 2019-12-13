import sklearn
#sklearn.show_versions()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# TODO ------------------------------------------------------------- update for each exp
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
# TODO ------------------------------------------------------------- update for each exp
model = GaussianNB()
model.fit(X_train, y_train.values.ravel()) # fix 3

# predict Output
predicted_new_bike_shares = model.predict(X_test)

# stats
print('--- comparison of real value and pred ---')
print(model)
print('    true:')
print(y_test)
print('\n    predicted:')
print(predicted_new_bike_shares)
print('-----------------------------------------')

acc = accuracy_score(y_test, predicted_new_bike_shares)
f1 = f1_score(y_test, predicted_new_bike_shares, average='macro')
mae = mean_absolute_error(y_test, predicted_new_bike_shares)
mse = mean_squared_error(y_test, predicted_new_bike_shares)
pr = precision_score(y_test, predicted_new_bike_shares, average='macro')
r2 = r2_score(y_test, predicted_new_bike_shares)
rec = recall_score(y_test, predicted_new_bike_shares, average='macro')
metrics_msg = 'accuracy: {} - f1: {} - mean absolute errror: {} - mean squared error: {} \n - precision: {} - r2 score: {} - recall: {} \n \n'.format(acc, f1, mae, mse, pr, r2, rec)
# "/home/kacham/Documents/tracelogs/metrics/(exp)_(challenge)_(model)_buggy/corrected_metrics.txt"
# TODO ------------------------------------------------------------- update for each exp
with open("/home/kacham/Documents/tracelogs/metrics/(exp)_london_merged_(model)_(eval)_metrics.txt", "a") as myfile:
    myfile.write(metrics_msg)
with open("/home/kacham/Documents/tracelogs/params/(exp)_london_merged_(model)_(eval)_params.txt", "a") as myfile:
    print(model,file=myfile)
    myfile.write('True:')
    print(y_test,file=myfile)
    myfile.write('Predicted:')
    print(predicted_new_bike_shares,file=myfile)
    myfile.write('\n' '\n')

_________________________________
continuous :

import sklearn
#sklearn.show_versions()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# TODO ------------------------------------------------------------- update for each exp
from sklearn.linear_model import BayesianRidge

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# load data
dataset_url = '/home/kacham/Documents/datasets_kaggle/continuous/preprocessed_data.csv'
data = pd.read_csv(dataset_url)

# separate features
y_features = ['B_avg_LEG_landed', 'R_avg_LEG_landed']
y = data[y_features]
X_features = ['B_Height_cms', 'R_Height_cms', 'B_avg_SIG_STR_landed', 'R_avg_SIG_STR_landed', 'B_avg_opp_SIG_STR_landed', 'R_avg_opp_SIG_STR_landed', 'B_avg_GROUND_landed', 'R_avg_GROUND_landed', 'B_avg_opp_GROUND_landed', 'R_avg_opp_GROUND_landed', 'B_avg_opp_LEG_landed', 'R_avg_opp_LEG_landed']
X = data[X_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123) #fix 2, no stratify

# train a Gaussian classifier
# TODO ------------------------------------------------------------- update for each exp
model = BayesianRidge()
model.fit(X_train, y_train.values.ravel()) # fix 3

# predict Output
predicted_health_life_expectancy = model.predict(X_test)

# stats
print('--- comparison of real value and pred ---')
print(model)
print('    true:')
print(y_test)
print('\n    predicted:')
print(predicted_health_life_expectancy)
print('-----------------------------------------')

evs = explained_variance_score(y_test, predicted_health_life_expectancy)
mae = mean_absolute_error(y_test, predicted_health_life_expectancy)
mse = mean_squared_error(y_test, predicted_health_life_expectancy)
r2 = r2_score(y_test, predicted_health_life_expectancy)
metrics_msg = 'explained variance score: {} - mean absolute errror: {} - mean squared error: {} - r2 score: {} \n \n'.format(evs, mae, mse, r2)
# "/home/kacham/Documents/tracelogs/metrics/(exp)_(challenge)_(model)_buggy/corrected_metrics.txt"
# TODO ------------------------------------------------------------- update for each exp
with open("/home/kacham/Documents/tracelogs/metrics/(exp)_world-happiness_(model)_(eval)_metrics.txt", "a") as myfile:
    myfile.write(metrics_msg)
with open("/home/kacham/Documents/tracelogs/params/(exp)_world-happiness_(model)_(eval)_params.txt", "a") as myfile:
    print(model,file=myfile)
    myfile.write('True:')
    print(y_test,file=myfile)
    myfile.write('Predicted:')
    print(predicted_health_life_expectancy,file=myfile)
    myfile.write('\n' '\n')
