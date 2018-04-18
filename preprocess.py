import pandas as pd
import seaborn
import numpy as np

seaborn.set()
np.set_printoptions(precision=3, linewidth=100, suppress=True)

df = pd.read_csv('data/race-result-horse.csv',
                 na_values=['DISQ', 'DNF', 'FE', 'PU', 'TNP', 'UR', 'VOID', 'WR', 'WV', 'WV-A', 'WX', 'WX-A', 'nan'])

df = df.dropna(subset=['finishing_position'])

df['finishing_position'] = df['finishing_position'].replace([' DH'], '', regex=True)

# Add a column named recent_6_runs to the dataframe, which records the recent ranks of the horse in
#each entry. The ranks are separated by “/”, and a record is like 1/2/6/3/4/7.
id = []
for item in df['horse_id']:
    if item not in id:
        id.append(item)
print (len(id))




#Add a column named recent_ave_rank for each entry to the dataframe, which records the average
#rank of the recent 6 runs of a horse. If there are less than 6 past runs, take the average of all the past
#runs. For example, the horse with past ranks 3,5,13 has a average rank (3+5+13)/3=7. If there are
#no previous runs, set the recent_ave_rank to be a prior value 7.

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


