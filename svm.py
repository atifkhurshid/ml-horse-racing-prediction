from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn import svm
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


####################################################load data########################################################
global Xtrain, kfold

df_train = pd.read_csv('data/training.csv')
df_test = pd.read_csv('data/testing.csv')

data_train = df_train[['jockey_ave_rank', 'trainer_ave_rank', 'actual_weight', 'declared_horse_weight',
                       'draw', 'win_odds', 'recent_ave_rank', 'race_distance', 'horse_win',
                       'horse_rank_top_3', 'horse_rank_top_50_percent']].values
data_test = df_test[['jockey_ave_rank', 'trainer_ave_rank', 'actual_weight', 'declared_horse_weight',
                     'draw', 'win_odds', 'recent_ave_rank', 'race_distance', 'horse_win',
                     'horse_rank_top_3', 'horse_rank_top_50_percent']].values

kfold = KFold(n_splits=10)

def find_slice(index, target):
    i = 0
    for train_index, test_index in kfold.split(Xtrain):
        if i < index:
            pass
        else:
            X_train = Xtrain[train_index]
            y_train = target[train_index]
            break
        i = i + 1
    return X_train, y_train

Ytrain_HorseWin, Ytrain_HorseRankTop3, Ytrain_HorseRankTop50Percent = data_train[-5000:,8:9].ravel(), data_train[-5000:,9:10].ravel(), data_train[-5000:,10:11].ravel()
Ytest_HorseWin, Ytest_HorseRankTop3, Ytest_HorseRankTop50Percent = data_test[-5000:,8:9].ravel(), data_test[-5000:,9:10].ravel(), data_test[-5000:,10:11].ravel()

train_y = [Ytrain_HorseWin, Ytrain_HorseRankTop3, Ytrain_HorseRankTop50Percent]
test_y = [Ytest_HorseWin, Ytest_HorseRankTop3, Ytest_HorseRankTop50Percent]

scores_svm_1 = {}
scores_svm_3 = {}
scores_svm_50 = {}
scores_svm = [scores_svm_1, scores_svm_3, scores_svm_50]

for train, test, scores in zip(train_y, test_y, scores_svm):
    for i in range(2, 8):
        combins = [c for c in  combinations(range(8), i)]
        for features in combins:

            Xtrain = np.asarray([data_train[-5000:, i] for i in features]).T
            Xtest = np.asarray([data_test[-5000:, i] for i in features]).T
        
            X_scaler = StandardScaler()
            Xtrain = X_scaler.fit_transform(Xtrain)
            Xtest = X_scaler.transform(Xtest)
     
            rbf_svc = svm.SVC(class_weight='balanced', kernel='rbf', C = 15000, gamma= 0.000001)
        
            #HorseWin
            results = cross_val_score(rbf_svc, Xtrain, train, cv=kfold, scoring='accuracy')
            index = np.argmax(results)
            X_train, y_train =  find_slice(index, train)

            rbf_svc.fit(X_train, y_train)
            predict = rbf_svc.predict(Xtest)
            scores[features] = f1_score(test, predict)

key_svm_1 = max(scores_svm_1, key=scores_svm_1.get)
key_svm_3 = max(scores_svm_3, key=scores_svm_3.get)
key_svm_50 = max(scores_svm_50, key=scores_svm_50.get)
print(key_svm_1, scores_svm_1[key_svm_1])
print(key_svm_3, scores_svm_3[key_svm_3])
print(key_svm_50, scores_svm_50[key_svm_50])
