from sklearn import linear_model
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from naive_bayes import NaiveBayes
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd


####################################################load data########################################################
df_train = pd.read_csv('/home/whl/Downloads/Horse-Racing-Prediction-master/data/training.csv')
df_test = pd.read_csv('/home/whl/Downloads/Horse-Racing-Prediction-master/data/testing.csv')

data_train = df_train[['jockey_ave_rank', 'trainer_ave_rank', 'actual_weight', 'declared_horse_weight',
                       'draw', 'win_odds', 'recent_ave_rank', 'race_distance', 'horse_win',
                       'horse_rank_top_3', 'horse_rank_top_50_percent']].values
data_test = df_test[['jockey_ave_rank', 'trainer_ave_rank', 'actual_weight', 'declared_horse_weight',
                     'draw', 'win_odds', 'recent_ave_rank', 'race_distance', 'horse_win',
                     'horse_rank_top_3', 'horse_rank_top_50_percent']].values

Xtrain, Ytrain_HorseWin, Ytrain_HorseRankTop3, Ytrain_HorseRankTop50Percent = data_train[:,0:8], data_train[:,8:9].ravel(), data_train[:,9:10].ravel(), data_train[:,10:11].ravel()
Xtest, Ytest_HorseWin, Ytest_HorseRankTop3, Ytest_HorseRankTop50Percent = data_test[:,0:8], data_test[:,8:9].ravel(), data_test[:,9:10].ravel(), data_test[:,10:11].ravel()
#kfold = KFold(n_splits=10)


###########################################logistic regression classifier###############################################
lr_model = linear_model.LogisticRegression()

t0 = datetime.now()
lr_model.fit(Xtrain, Ytrain_HorseWin)
t1 = datetime.now() - t0
print ("Training time for the first logistic regression classifier:", t1)
predict_HorseWin = lr_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseWin, predict_HorseWin)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,8:9], cv=kfold, scoring='accuracy')
#results_HorseWin = results.mean()

t0 = datetime.now()
lr_model.fit(Xtrain, Ytrain_HorseRankTop3)
t2 = datetime.now() - t0
print ("Training time for the second logistic regression classifier:", t2)
predict_HorseRankTop3 = lr_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop3, predict_HorseRankTop3)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,9:10], cv=kfold, scoring='accuracy')
#results_HorseRankTop3 = results.mean()

t0 = datetime.now()
lr_model.fit(Xtrain, Ytrain_HorseRankTop50Percent)
t3 = datetime.now() - t0
print ("Training time for the third logistic regression classifier:", t3)
predict_HorseRankTop50Percent = lr_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop50Percent, predict_HorseRankTop50Percent)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,10:11], cv=kfold, scoring='accuracy')
#results_HorseRankTop50Percent = results.mean()

print ("Average training time for logistic regression classifier:", (t1+t2+t3)/3)


#################################################naive bayes classifier##################################################
nb_model = NaiveBayes()
gnb = GaussianNB()

t0 = datetime.now()
nb_model.fit(Xtrain, Ytrain_HorseWin)
t11 = datetime.now() - t0
print ("Training time for my own NaiveBayes classifier:", t11)

t0 = datetime.now()
gnb.fit(Xtrain, Ytrain_HorseWin)
t12 = datetime.now() - t0
print ("Training time for sklearn NaiveBayes classifier:", t12)

predict_HorseWin = nb_model.predict(Xtest)
print('Average precision-recall score for my own Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseWin, predict_HorseWin)))
print('Average precision-recall score for sklearn Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseWin, gnb.predict(Xtest))))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,8:9], cv=kfold, scoring='accuracy')
#results_HorseWin = results.mean()

t0 = datetime.now()
nb_model.fit(Xtrain, Ytrain_HorseRankTop3)
t21 = datetime.now() - t0
print ("Training time for my own NaiveBayes classifier:", t21)

t0 = datetime.now()
gnb.fit(Xtrain, Ytrain_HorseRankTop3)
t22 = datetime.now() - t0
print ("Training time for sklearn NaiveBayes classifier:", t22)

predict_HorseRankTop3 = nb_model.predict(Xtest)
print('Average precision-recall score for my own Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop3, predict_HorseRankTop3)))
print('Average precision-recall score for sklearn Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop3, gnb.predict(Xtest))))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,9:10], cv=kfold, scoring='accuracy')
#results_HorseRankTop3 = results.mean()

t0 = datetime.now()
nb_model.fit(Xtrain, Ytrain_HorseRankTop50Percent)
t31 = datetime.now() - t0
print ("Training time for my own NaiveBayes classifier:", t31)

t0 = datetime.now()
gnb.fit(Xtrain, Ytrain_HorseRankTop50Percent)
t32 = datetime.now() - t0
print ("Training time for sklearn NaiveBayes classifier:", t32)

predict_HorseRankTop50Percent = nb_model.predict(Xtest)
print('Average precision-recall score for my own Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop50Percent, predict_HorseRankTop50Percent)))
print('Average precision-recall score for sklearn Gaussian Naive Bayes classifier: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop50Percent, gnb.predict(Xtest))))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,10:11], cv=kfold, scoring='accuracy')
#results_HorseRankTop50Percent = results.mean()

print ("Average training time for my own NaiveBayes classifier:", (t11+t21+t31)/3)
print ("Average training time for sklearn NaiveBayes classifier:", (t12+t22+t32)/3)


######################################################svm classifier#######################################################
svm_model = svm.SVC(kernel='linear')

t0 = datetime.now()
svm_model.fit(Xtrain, Ytrain_HorseWin)
t1 = datetime.now() - t0
print ("Training time for the first logistic regression classifier:", t1)
predict_HorseWin = svm_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseWin, predict_HorseWin)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,8:9], cv=kfold, scoring='accuracy')
#results_HorseWin = results.mean()

t0 = datetime.now()
svm_model.fit(Xtrain, Ytrain_HorseRankTop3)
t2 = datetime.now() - t0
print ("Training time for the second logistic regression classifier:", t2)
predict_HorseRankTop3 = svm_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop3, predict_HorseRankTop3)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,9:10], cv=kfold, scoring='accuracy')
#results_HorseRankTop3 = results.mean()

t0 = datetime.now()
svm_model.fit(Xtrain, Ytrain_HorseRankTop50Percent)
t3 = datetime.now() - t0
print ("Training time for the third logistic regression classifier:", t3)
predict_HorseRankTop50Percent = svm_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop50Percent, predict_HorseRankTop50Percent)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,10:11], cv=kfold, scoring='accuracy')
#results_HorseRankTop50Percent = results.mean()

print ("Average training time for logistic regression classifier:", (t1+t2+t3)/3)


######################################################random forest classifier#######################################################
rf_model = RandomForestClassifier(max_depth=2, random_state=0)

t0 = datetime.now()
rf_model.fit(Xtrain, Ytrain_HorseWin)
t1 = datetime.now() - t0
print ("Training time for the first logistic regression classifier:", t1)
predict_HorseWin = rf_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseWin, predict_HorseWin)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,8:9], cv=kfold, scoring='accuracy')
#results_HorseWin = results.mean()

t0 = datetime.now()
rf_model.fit(Xtrain, Ytrain_HorseRankTop3)
t2 = datetime.now() - t0
print ("Training time for the second logistic regression classifier:", t2)
predict_HorseRankTop3 = rf_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop3, predict_HorseRankTop3)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,9:10], cv=kfold, scoring='accuracy')
#results_HorseRankTop3 = results.mean()

t0 = datetime.now()
rf_model.fit(Xtrain, Ytrain_HorseRankTop50Percent)
t3 = datetime.now() - t0
print ("Training time for the third logistic regression classifier:", t3)
predict_HorseRankTop50Percent = rf_model.predict(Xtest)
print('Average precision-recall score: {0:0.5f}'.format(f1_score(Ytest_HorseRankTop50Percent, predict_HorseRankTop50Percent)))
#results = cross_val_score(lr_model, data_train[:,0:8], data_train[:,10:11], cv=kfold, scoring='accuracy')
#results_HorseRankTop50Percent = results.mean()

print ("Average training time for logistic regression classifier:", (t1+t2+t3)/3)
