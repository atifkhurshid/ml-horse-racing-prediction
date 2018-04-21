import pandas as pd
import seaborn

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

path = 'predictions/'
lr_file = pd.read_csv(path + 'lr_predictions.csv')
nb_file = pd.read_csv(path + 'nb_predictions.csv')
svm_file = pd.read_csv(path + 'svm_predictions.csv')
rf_file = pd.read_csv(path + 'rf_predictions.csv')
rg_file = pd.read_csv(path + 'gbrt_predictions.csv')

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


weights = {'lr':0, 'nb':0, 'svm':0, 'rf':50, 'gbrt':50} #1009

returns = calculate_returns(df_test, dataframe, weights, 10, num_races)

print ("Balance : ", returns)
print("Money won: ",returns - num_races)
print ("Won: ", won, "Lost: ",lost)
print("Win loss ratio: ", won/lost)