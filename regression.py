import pandas as pd
import seaborn
import math
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
seaborn.set()

def Time_to_label(df_test, y_pred):
    top1 = np.zeros(y_pred.shape)
    top3 = np.zeros(y_pred.shape)
    top50 = np.zeros(y_pred.shape)
    current_race_ID = df_test.loc[0].race_id
    race_time = []
    indices = []
    for index, row in df_test.iterrows():
        if current_race_ID == row.race_id:
            race_time.append(y_pred[index])
            indices.append(index)
        else:
            index_array = [x for _, x in sorted(zip(race_time, indices))]
            top1[index_array[0]] = 1
            top3[index_array[0]] = 1
            top3[index_array[1]] = 1
            top3[index_array[2]] = 1
            size = len(index_array)
            count = 1
            for c in index_array:
                if count / size <= 0.5:
                    top50[c] = 1
                else:
                    break
                count += 1
            indices = [index]
            race_time = [y_pred[index]]
        current_race_ID = row.race_id
    return top1, top3, top50

def write_csv(a, b, c, race_id, horse_id, path, name):
    columns = ['RaceID','HorseID','HorseWin','HorseRankTop3','HorseRankTop50Percent']
    dataframe = pd.DataFrame({'RaceID':race_id, 'HorseID':horse_id, 'HorseWin':a, 'HorseRankTop3':b, 'HorseRankTop50Percent':c})
    dataframe.to_csv("{0}/{1}_predictions.csv".format(path, name), index=False, sep=',', columns=columns)


def Top_1_3_avg(df_test, y_pred):
    current_race_ID = df_test.loc[0].race_id
    race_time = []
    indices = []
    num_races = 1
    top1 = 0
    top3 = 0
    avg_rank = 0
    for index, row in df_test.iterrows():
        if current_race_ID == row.race_id:
            race_time.append(y_pred[index])
            indices.append(index)
        else:
            num_races += 1
            index_array = [x for _, x in sorted(zip(race_time, indices))]
            top_pos = int(df_test.loc[index_array[0]].finishing_position)
            avg_rank += top_pos
            if top_pos == 1:
                top1 += 1
            if top_pos <= 3:
                top3 += 1
            indices = [index]
            race_time = [y_pred[index]]
        current_race_ID = row.race_id
    return float(top1) / num_races, float(top3) / num_races, float(avg_rank) / num_races


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


std_scalar = StandardScaler()
std_scalar.fit(X_train)
X_train_std = std_scalar.transform(X_train)
X_test_std = std_scalar.transform(X_test)
joblib.dump(std_scalar, 'models/std_scalar.pkl')

std_scalar_y = StandardScaler()
std_scalar_y.fit(np.reshape(y_train, (-1, 1)))
y_train_std = std_scalar_y.transform(np.reshape(y_train, (-1, 1))).ravel()
joblib.dump(std_scalar_y, 'models/std_scalar_y.pkl')

#model = SVR(kernel='linear', C = 0.1, epsilon=1)         #1.62 without normal
#model = SVR(kernel='linear', C = 10) #1.59 with full data - normalized
#model = SVR(kernel='rbf', C = 5000, gamma= 0.0000001)     #1.561 with all data
#model = SVR(kernel='rbf', C = 1000000, gamma= 0.000001)     #1.592 with all data - normalized
#model = SVR(kernel='sigmoid', C = 1, gamma=0.01) #1.61 with full data - normalized
#model = SVR(kernel = 'poly', degree=2, C=0.00001) #1.8
#model = SVR(kernel = 'poly', degree=3, coef0=1) #1.565 full data - normalized

#Reason : Fastest to train, same performance as others

#svr_model = SVR(kernel='rbf', C=5000, epsilon=0.1, gamma=0.0000001)        1.59 .227 .496 4.619


print ("model_name          RMSE   Top_1   Top_3     Average_Rank")
svr_model = SVR(kernel='rbf', C = 5000, epsilon=0.01, gamma= 0.000001)
svr_model.fit(X_train, y_train)
joblib.dump(svr_model, 'models/svr_model.pkl')
svr_pred = svr_model.predict(X_test)
svr_RMSE = math.sqrt(mean_squared_error(y_test, svr_pred))
svr_top1, svr_top3, svr_avg = Top_1_3_avg(df_test, svr_pred)
print ("svr_model,         %.3f   %.3f    %.3f       %.3f" %(svr_RMSE, svr_top1, svr_top3, svr_avg))

s_svr_model = SVR(kernel='rbf', C = 1000, epsilon=0.01, gamma= 0.0001)
s_svr_model.fit(X_train_std, y_train_std)
joblib.dump(s_svr_model, 'models/s_svr_model.pkl')
s_svr_pred = s_svr_model.predict(X_test_std)
s_svr_pred = std_scalar_y.inverse_transform(s_svr_pred)
s_svr_RMSE = math.sqrt(mean_squared_error(y_test, s_svr_pred))
s_svr_top1, s_svr_top3, s_svr_avg = Top_1_3_avg(df_test, s_svr_pred)
print ("Scaled svr_model,  %.3f   %.3f    %.3f       %.3f" %(s_svr_RMSE, s_svr_top1, s_svr_top3, s_svr_avg))

gbrt_model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=120, max_depth=5, random_state=42)
gbrt_model.fit(X_train, y_train)
joblib.dump(gbrt_model, 'models/gbrt_model.pkl')
gbrt_pred = gbrt_model.predict(X_test)
gbrt_RMSE = math.sqrt(mean_squared_error(y_test, gbrt_pred))
gbrt_top1, gbrt_top3, gbrt_avg = Top_1_3_avg(df_test, gbrt_pred)

print ("gbrt_model,        %.3f   %.3f    %.3f       %.3f" %(gbrt_RMSE, gbrt_top1, gbrt_top3, gbrt_avg))


s_gbrt_model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=120, max_depth=5, random_state=42)
s_gbrt_model.fit(X_train_std, y_train_std)
joblib.dump(s_gbrt_model, 'models/s_gbrt_model.pkl')
s_gbrt_pred = s_gbrt_model.predict(X_test_std)
s_gbrt_pred = std_scalar_y.inverse_transform(s_gbrt_pred)
s_gbrt_RMSE = math.sqrt(mean_squared_error(y_test, s_gbrt_pred))
s_gbrt_top1, s_gbrt_top3, s_gbrt_avg = Top_1_3_avg(df_test, s_gbrt_pred)
print ("Scaled gbrt_model, %.3f   %.3f    %.3f       %.3f" %(s_gbrt_RMSE, s_gbrt_top1, s_gbrt_top3, s_gbrt_avg))
