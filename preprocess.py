import pandas as pd
import seaborn
import numpy as np

seaborn.set()
np.set_printoptions(precision=3, linewidth=100, suppress=True)

df = pd.read_csv('data/race-result-horse.csv',
                 na_values=['DISQ', 'DNF', 'FE', 'PU', 'TNP', 'UR', 'VOID', 'WR', 'WV', 'WV-A', 'WX', 'WX-A', 'nan'])

df = df.dropna(subset=['finishing_position'])

df['finishing_position'] = df['finishing_position'].replace([' DH'], '', regex=True)


df['recent_6_runs'] = 'nan'
df['recent_ave_rank'] = 7
id_list = []
for id in df['horse_id']:
    if id not in id_list:
        id_list.append(id)
for id in id_list:
    breaker = '/'
    recent = []
    for index, record in df[df['horse_id'] == id].iterrows():
        df.loc[index, 'recent_6_runs'] = breaker.join(recent)
        if (len(recent)) != 0:
            df.loc[index, 'recent_ave_rank'] = sum(map(float,recent))/float(len(recent))
        recent.append(record.finishing_position)
        if len(recent) > 6:
            recent = recent[1:]

print(df)


# Give a unique index to each horses, where “which horse has which index” is not restricted. Similarly, a
#unique index should be assigned to each jockey, and a unique index should be assigned to each trainer.


#Add a column named jockey_ave_rank to the dataframe that records the average rank of the jockey
#in the training data. Similarly, add a column named trainer_ave_rank that records the average rank
#of the trainer in the training data. Note that if a jockey or a trainer doesn’t appear in the training
#data, set the average rank of the jockey to be 7.

# Read the distance information in race-result-race.csv and add a column to the dataframe race_distance
#for each entry in race-result-horse.csv.


# ADD THREE COLUMNS Top1, Top3 or TOp 50% - 1/0

#Split the dataframe after all the above pre-processing into two parts, and save them to files. The
#training data should be saved as training.csv and the testing data be saved as testing.csv.


