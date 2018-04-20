from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


####################################################load data########################################################
global Xtrain, kfold

df_train = pd.read_csv('/home/whl/Downloads/Horse-Racing-Prediction-master/data/training.csv')
df_test = pd.read_csv('/home/whl/Downloads/Horse-Racing-Prediction-master/data/testing.csv')

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

scores_lr = {}
scores_gnb = {}
scores_rf = {}
for i in range(2, 8):
    combins = [c for c in  combinations(range(8), i)]
    for features in combins:

        Xtrain, Ytrain_HorseWin, Ytrain_HorseRankTop3, Ytrain_HorseRankTop50Percent = np.asarray([data_train[:, i] for i in features]).T, data_train[:,8:9].ravel(), data_train[:,9:10].ravel(), data_train[:,10:11].ravel()
        Xtest, Ytest_HorseWin, Ytest_HorseRankTop3, Ytest_HorseRankTop50Percent = np.asarray([data_test[:, i] for i in features]).T, data_test[:,8:9].ravel(), data_test[:,9:10].ravel(), data_test[:,10:11].ravel()
        
        X_scaler = StandardScaler()
        Xtrain = X_scaler.fit_transform(Xtrain)
        Xtest = X_scaler.transform(Xtest)

        ############################################logistic regression classifier#############################################
        lr_model = linear_model.LogisticRegression()
        
        #HorseWin
        results = cross_val_score(lr_model, Xtrain, Ytrain_HorseWin, cv=kfold, scoring='accuracy')
        index = np.argmax(results)
        X_train, y_train =  find_slice(index, Ytrain_HorseWin)

        lr_model.fit(X_train, y_train)
        predict_HorseWin = lr_model.predict(Xtest)
        scores_lr[features] = f1_score(Ytest_HorseWin, predict_HorseWin)

        gnb = GaussianNB()
        
        #HorseWin
        results = cross_val_score(gnb, Xtrain, Ytrain_HorseWin, cv=kfold, scoring='accuracy')
        index = np.argmax(results)
        X_train, y_train =  find_slice(index, Ytrain_HorseWin)

        gnb.fit(X_train, y_train)
        predict_HorseWin = gnb.predict(Xtest)
        scores_gnb[features] = f1_score(Ytest_HorseWin, predict_HorseWin)
        
        rf_model = RandomForestClassifier(max_depth=8, random_state=0)
        
        #HorseWin
        results = cross_val_score(rf_model, Xtrain, Ytrain_HorseWin, cv=kfold, scoring='accuracy')
        index = np.argmax(results)
        X_train, y_train =  find_slice(index, Ytrain_HorseWin)

        rf_model.fit(X_train, y_train)
        predict_HorseWin = rf_model.predict(Xtest)
        scores_rf[features] = f1_score(Ytest_HorseWin, predict_HorseWin)


key_lr = max(scores_lr, key=scores_lr.get)
key_gnb = max(scores_gnb, key=scores_gnb.get)
key_rf = max(scores_rf, key=scores_rf.get)
print(key_lr, scores_lr[key_lr])
print(key_gnb, scores_gnb[key_gnb])
print(key_rf, scores_rf[key_rf])



df_train = pd.read_csv('data/training.csv')
df_test = pd.read_csv('data/testing.csv')

features = ['jockey_ave_rank', 'trainer_ave_rank', 'actual_weight', 'declared_horse_weight',
                       'draw', 'win_odds', 'recent_ave_rank', 'race_distance']

data_train = df_train[features].values
data_test = df_test[features].values

top1_train = df_train['horse_win'].values
top3_train = df_train['horse_rank_top_3'].values
top50_train = df_train['horse_rank_top_50_percent'].values

top1_test = df_test['horse_win'].values
top3_test = df_test['horse_rank_top_3'].values
top50_test = df_test['horse_rank_top_50_percent'].values


lr = linear_model.LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(data_train, top1_train)

train_pred = lr.predict(data_train)
test_pred = lr.predict(data_test)

print (classification_report(top1_train, train_pred))
print (confusion_matrix(top1_train, train_pred))
print (classification_report(top1_test, test_pred))
print (confusion_matrix(top1_test, test_pred))


rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(data_train, top1_train)

train_pred = rf.predict(data_train)
test_pred = rf.predict(data_test)

print (classification_report(top1_train, train_pred))
print (confusion_matrix(top1_train, train_pred))
print (classification_report(top1_test, test_pred))
print (confusion_matrix(top1_test, test_pred))
