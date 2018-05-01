import pandas as pd
import seaborn
import numpy as np
from sklearn.externals import joblib

seaborn.set()

won = 0
lost = 0

def bet(df_test, index, bet_amount):
    global won, lost
    if df_test.loc[index].finishing_position == 1:
        won += 1
        return bet_amount * (df_test.loc[index].win_odds)
    else:
        lost += 1
        return -bet_amount


def calculate_score (row, weight_dict, win_odds=1):
    return (weight_dict['lr']*int(row.lr) + weight_dict['nb']*int(row.nb)
                         + weight_dict['svm']*int(row.svm) + weight_dict['rf']*int(row.rf)
                         + weight_dict['svr']*int(row.svr) + weight_dict['s_svr']*int(row.s_svr)
                         + weight_dict['gbrt']*int(row.gbrt) + weight_dict['gbrt_s']*int(row.s_gbrt))/ float(win_odds)


def calculate_returns(df_test, df_predictions, weight_dict, threshold, bet_amount, strategy = False):
    current_race_ID = df_predictions.loc[0].RaceID
    indices = []
    score = []
    money = 0
    for index, row in df_predictions.iterrows():
        if current_race_ID == row.RaceID:
            if strategy:
                res = calculate_score(row, weight_dict, df_test.loc[index].win_odds)
            else:
                res = calculate_score(row, weight_dict)
            score.append(res)
            indices.append(index)
        else:
            index_array = [x for _, x in sorted(zip(score, indices), reverse=True)]
            max_score = max(score)
            if max_score > threshold:
                money += bet(df_test, index_array[0], bet_amount)
            else:
                money += bet_amount  # Keep the money if no bet
            if strategy:
                res = calculate_score(row, weight_dict, df_test.loc[index].win_odds)
            else:
                res = calculate_score(row, weight_dict)
            score = [res]
        current_race_ID = row.RaceID
    return money


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


def get_result(file):
    result = []
    i = 0
    for race_id in race_list:
        dataframe = file[file['RaceID'] == race_id]
        horses = dataframe['HorseID']
        horses = horses.values.tolist()
        for horse_id in horses:
            data = dataframe[dataframe['HorseID'] == horse_id][features].values[0]
            result.append(max(data[0], data[1], data[2]))
            i = i + 1
    return result


print("Loading data...")
df_test = pd.read_csv("data/testing.csv")
features1 = ['actual_weight', 'declared_horse_weight','draw',
            'win_odds', 'jockey_ave_rank', 'trainer_ave_rank',
            'recent_ave_rank', 'race_distance']

X_test = np.array(df_test[features1])


path = 'predictions/'
lr_file = pd.read_csv(path + 'lr_predictions.csv')
nb_file = pd.read_csv(path + 'nb_predictions.csv')
svm_file = pd.read_csv(path + 'svm_predictions.csv')
rf_file = pd.read_csv(path + 'rf_predictions.csv')

features = ['HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

race_list = []
for id in lr_file['RaceID']:
    if id not in race_list:
        race_list.append(id)
num_races = len(race_list)

std_scalar = joblib.load("models/std_scalar.pkl")
std_scalar_y = joblib.load("models/std_scalar_y.pkl")

svr_model = joblib.load("models/svr_model.pkl")
s_svr_model = joblib.load("models/s_svr_model.pkl")
gbrt_model = joblib.load("models/gbrt_model.pkl")
s_gbrt_model = joblib.load("models/s_gbrt_model.pkl")

svr_pred = svr_model.predict(X_test)
s_svr_pred = std_scalar_y.inverse_transform(s_svr_model.predict(std_scalar.transform(X_test)))
gbrt_pred = gbrt_model.predict(X_test)
s_gbrt_pred = std_scalar_y.inverse_transform(s_gbrt_model.predict(std_scalar.transform(X_test)))

svr_top1 , _, _ = Time_to_label(df_test, svr_pred)
s_svr_top1 , _, _ = Time_to_label(df_test, s_svr_pred)
gbrt_top1, _, _ = Time_to_label(df_test, gbrt_pred)
s_gbrt_top1, _, _ = Time_to_label(df_test, s_gbrt_pred)


reult_lr = get_result(lr_file)
reult_nb = get_result(nb_file)
reult_svm = get_result(svm_file)
reult_rf = get_result(rf_file)
print ("Results ready!")
race_id = lr_file['RaceID'].values
horse_id = lr_file['HorseID'].values

dataframe = pd.DataFrame( {'RaceID': race_id, 'HorseID': horse_id, 'lr': reult_lr,
                           'nb': reult_nb, 'svm': reult_svm, 'rf': reult_rf,'svr': svr_top1,
                            's_svr': s_svr_top1, 'gbrt': gbrt_top1 ,'s_gbrt': s_gbrt_top1})

weights = {'lr':15, 'nb':26, 'svm':15, 'rf':22, 'svr':0, 's_svr':0, 'gbrt':0, 'gbrt_s':22}

print ("Number of races: ", num_races)
print ("Bet per race: $1")

returns = calculate_returns(df_test, dataframe, weights, threshold=0, bet_amount=1, strategy = False)
print("Money won by default strategy: ",returns)
print("Won: %d, Lost: %d, WL-ratio: %.2f" %(won, lost, float(won)/lost))
won = 0
lost = 0
returns = calculate_returns(df_test, dataframe, weights, threshold=35, bet_amount=1, strategy = True)
print("Money won by our strategy: ",returns)
print("Won: %d, Lost: %d, WL-ratio: %.2f" %(won, lost, float(won)/lost))
