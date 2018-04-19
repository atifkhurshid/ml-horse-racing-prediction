import pandas as pd
import seaborn
import math
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

seaborn.set()
np.set_printoptions(precision=3, suppress=True, linewidth= 200 , threshold=1000)

print("Loading data...")
df_train = pd.read_csv('data/training.csv')
df_test = pd.read_csv('data/testing.csv')

features = ['actual_weight', 'declared_horse_weight','draw',
            'win_odds', 'jockey_ave_rank', 'trainer_ave_rank',
            'recent_ave_rank', 'race_distance']

X_train = np.array(df_train[features])
X_test = np.array(df_test[features])

print("Processing timestamps...")
finish_time = df_train['finish_time']
y_train = []
for t in finish_time:
    t_arr = t.split('.')
    y_train.append(float(t_arr[0])*60 + float(t_arr[1] + '.' + t_arr[2] ))
y_train = np.array(y_train)

finish_time = df_test['finish_time']
y_test = []
for t in finish_time:
    t_arr = t.split('.')
    y_test.append(float(t_arr[0])*60 + float(t_arr[1] + '.' + t_arr[2] ))
y_test = np.array(y_test)


X_train = X_train[0:500]
y_train = y_train[0:500]

'''
std_scalar = StandardScaler()
std_scalar.fit(X_train)
X_train = std_scalar.transform(X_train)

'''


parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 30, 50, 70, 100 ],
               'verbose': True },
              {'kernel': ['linear'],
                'C': [1, 10, 30, 50, 70, 100 ],
               'verbose': True},
              {'kernel': ['sigmoid'],
                'C': [1, 10, 30, 50, 70, 100 ],
               'verbose': True},
              {'kernel': ['poly'],
               'C': [1, 10, 30, 50, 70, 100],
                'degree': [2, 3, 4, 5, 6, 7, 8],
               'verbose': True}
              ]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(SVR(), parameters, cv=5, scoring='explained_variance')
clf.fit(X_train[0:500], y_train[0:500])

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()



