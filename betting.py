import pandas as pd
import seaborn

seaborn.set()

def bet(df_test, index):
    if df_test.loc[index].finishing_position == 1:
        return float(df_test.loc[index].win_odds)
    else:
        return -1

def calculate_returns(df_test, df_predictions, weight_dict, threshold):
    current_race_ID = df_predictions.loc[0].race_id
    indices = []
    score = []
    returns = 0
    for index, row in df_predictions.iterrows():
        if current_race_ID == row.race_id:
            score.append((weight_dict['lr']*int(row.lr) + weight_dict['nb']*int*(row.nb)
                         + weight_dict['svm']*int(row.svm) + weight_dict['rf']*int(row.rf)
                         + weight_dict['gbrt']*int(row.gbrt))
                         /float(df_test.loc[index].win_odds))
            indices.append(index)
        else:
            index_array = [x for _, x in sorted(zip(score, indices), reverse=True)]
            max_score = max(score)
            if max_score > threshold:
                returns += bet(df_test, index_array[0])
            indices = [index]
            score = [(weight_dict['lr']*int(row.lr) + weight_dict['nb']*int*(row.nb)
                         + weight_dict['svm']*int(row.svm) + weight_dict['rf']*int(row.rf)
                         + weight_dict['gbrt']*int(row.gbrt))
                         /float(df_test.loc[index].win_odds)]
        current_race_ID = row.race_id
    return returns

print("Loading data...")
df_test = pd.read_csv("data/testing.csv")
df_predicions = pd.read_csv("predictions/predictions.csv")

weights = {'lr':0.2, 'nb':0.2, 'svm':0.2, 'rf':0.2, 'gbrt':0.2}
print (calculate_returns(df_test, df_predicions, weights, threshold=0))
