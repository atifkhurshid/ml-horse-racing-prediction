import seaborn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

seaborn.set()
won = 0
lost = 0
def bet(df_test, index):
    global won, lost
    if df_test.loc[index].finishing_position == 1:
        won+= 1
        return float(df_test.loc[index].win_odds)

    else:
        lost+=1
        return -1

def calculate_returns(df_test, df_predictions, weight_dict, threshold, money):
    current_race_ID = df_predictions.loc[0].RaceID
    indices = []
    score = []
    money = money
    for index, row in df_predictions.iterrows():
        if current_race_ID == row.RaceID:
            score.append((weight_dict['lr']*int(row.lr) + weight_dict['nb']*int(row.nb)
                         + weight_dict['svm']*int(row.svm) + weight_dict['rf']*int(row.rf)
                         + weight_dict['gbrt']*int(row.rg)) / float(df_test.loc[index].win_odds))
            indices.append(index)
        else:
            index_array = [x for _, x in sorted(zip(score, indices), reverse=True)]
            max_score = max(score)
            if max_score > threshold:
                money += bet(df_test, index_array[0])

            score = [(weight_dict['lr']*int(row.lr) + weight_dict['nb']*int(row.nb)
                         + weight_dict['svm']*int(row.svm) + weight_dict['rf']*int(row.rf)
                         + weight_dict['gbrt']*int(row.rg))/float(df_test.loc[index].win_odds)]
        current_race_ID = row.RaceID
    return money

print("Loading data...")
df_test = pd.read_csv("data/testing.csv")

lr_file = pd.read_csv(path + '/lr_predictions.csv')
nb_file = pd.read_csv(path + '/nb_predictions.csv')
svm_file = pd.read_csv(path + '/svm_predictions.csv')
rf_file = pd.read_csv(path + '/rf_predictions.csv')

df_train = pd.read_csv(path + '/data/training.csv')
df_test = pd.read_csv(path + '/data/testing.csv')

features1 = ['actual_weight', 'declared_horse_weight','draw',
            'win_odds', 'jockey_ave_rank', 'trainer_ave_rank',
            'recent_ave_rank', 'race_distance']

X_train = np.array(df_train[features1])
X_test = np.array(df_test[features1])

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

std_scalar_y = StandardScaler()
std_scalar_y.fit(np.reshape(y_train, (-1, 1)))
y_train_std = std_scalar_y.transform(np.reshape(y_train, (-1, 1))).ravel()

s_gbrt_model = GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=120, max_depth=5, random_state=42)
s_gbrt_model.fit(X_train_std, y_train_std)
s_gbrt_pred = s_gbrt_model.predict(X_test_std)
s_gbrt_pred = std_scalar_y.inverse_transform(s_gbrt_pred)

top1, top3, top50 = Time_to_label(df_test, s_gbrt_pred)
rg_file = pd.DataFrame({'RaceID':df_test['race_id'].values, 'HorseID':df_test['horse_id'].values, 'HorseWin':top1, 'HorseRankTop3':top3, 'HorseRankTop50Percent':top50})

features = ['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

race_list = []
for id in lr_file['RaceID']:
    if id not in race_list:
        race_list.append(id)
num_races = len(race_list)

def get_result(file):
    result = []
    i = 0
    for race_id in race_list:
        dataframe = file[file['RaceID'] == race_id]
        horses = dataframe['HorseID']
        horses = horses.values.tolist()
        for horse_id in horses:
            data = dataframe[dataframe['HorseID'] == horse_id][features].values[0]
            if data[0] + data[1] + data[2] == 3:
                result.append(1)
            else:
                result.append(0)
            i = i + 1
    return result


reult_lr = get_result(lr_file)
reult_nb = get_result(nb_file)
reult_svm = get_result(svm_file)
reult_rf = get_result(rf_file)
reult_rg = get_result(rg_file)
print ("Results ready!")
race_id = lr_file['RaceID'].values
horse_id = lr_file['HorseID'].values

columns = ['RaceID', 'HorseID', 'lr', 'nb', 'svm', 'rf', 'rg']
dataframe = pd.DataFrame(
    {'RaceID': race_id, 'HorseID': horse_id, 'lr': reult_lr, 'nb': reult_nb, 'svm': reult_svm, 'rf': reult_rf,
     'rg': reult_rg})

weights = {'lr':15, 'nb':26, 'svm':15, 'rf':22, 'gbrt':22} #1009

returns = calculate_returns(df_test, dataframe, weights, 10, num_races)

print ("Balance : ", returns)
print("Money won: ",returns - num_races)
print ("Won: ", won, "Lost: ",lost)
print("Win loss ratio: ", won/lost)
