import pandas as pd
import numpy as np

global features, race_list

path = '/predictions/'
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
        
def get_result(file):  
    result = []
    i = 0
    for race_id in race_list:
        dataframe = file[file['RaceID'] == race_id]
        horses = dataframe['HorseID']
        horses = horses.values.tolist()
        for horse_id in horses:
            data = dataframe[dataframe['HorseID'] == horse_id][features].values[0]
            if data[0] == 1:
                if data[1] == 1 or data[2] == 1:
                    result.append(1)
                else:
                    result.append(0)
            else:
                result.append(0)
            i = i + 1
    return result

reult_lr = get_result(lr_file)
reult_nb = get_result(nb_file)
reult_svm = get_result(svm_file)
reult_rf = get_result(rf_file)
reult_rg = get_result(rg_file)

race_id = lr_file['RaceID'].values
horse_id = lr_file['HorseID'].values
                      
columns = ['RaceID','HorseID', 'lr','nb', 'svm', 'rf', 'rg']
dataframe = pd.DataFrame({'RaceID':race_id, 'HorseID':horse_id, 'lr':reult_lr, 'nb':reult_nb, 'svm': reult_svm, 'rf':reult_rf, 'rg':reult_rg})
