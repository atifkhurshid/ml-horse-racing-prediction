import pandas as pd
import seaborn

seaborn.set()

print ('Reading race-result-horse.csv...')
df = pd.read_csv('data/race-result-horse.csv',
                 na_values=['DISQ', 'DNF', 'FE', 'PU', 'TNP', 'UR', 'VOID', 'WR', 'WV', 'WV-A', 'WX', 'WX-A', 'nan'])

print ('Reading race-result-race.csv...')
df2 = pd.read_csv('data/race-result-race.csv')

print ("Dropping non-integer finishing-positions...")
df = df.dropna(subset=['finishing_position'])
df['finishing_position'] = df['finishing_position'].replace([' DH'], '', regex=True)

print ("Creating new columns...")
df['recent_6_runs'] = 'NA'
df['recent_ave_rank'] = 7
df['horse_index'] = 'NA'
df['jockey_index'] = 'NA'
df['trainer_index'] = 'NA'
df['jockey_ave_rank'] = 7
df['trainer_ave_rank'] = 7
df['race_distance'] = 'NA'
df['race_course'] = 'NA'
df['race_class'] = 'NA'
df['track'] = 'NA'
df['track_condition'] = 'NA'
df['horse_win'] = 0
df['horse_rank_top_3'] = 0
df['horse_rank_top_50_percent'] = 0

print ("Generating lists of races, horses, jockeys and trainers...")
race_ids = []
id_list = []
jockey_list = []
trainer_list = []
idx = 0

for id in df['race_id'].values:
    if id not in race_ids:
        race_ids.append(id)

for id in df['horse_name'].values:
    if id not in id_list:
        id_list.append(id)

for name in df['jockey'].values:
    if name not in jockey_list:
        jockey_list.append(name)

for name in df['trainer'].values:
    if name not in trainer_list:
        trainer_list.append(name)

print ("Number of horses: ", len(id_list))
print ("Number of jockeys: ", len(jockey_list))
print ("Number of trainers: ", len(trainer_list))


print ("Adding racecourse information...")
for id in race_ids:
    df.loc[df.race_id == id, 'race_distance'] = df2.loc[df2.race_id == id, 'race_distance'].item()
    df.loc[df.race_id == id, 'race_course'] = df2.loc[df2.race_id == id, 'race_course'].item()
    df.loc[df.race_id == id, 'race_class'] = df2.loc[df2.race_id == id, 'race_class'].item()
    df.loc[df.race_id == id, 'track'] = df2.loc[df2.race_id == id, 'track'].item()
    df.loc[df.race_id == id, 'track_condition'] = df2.loc[df2.race_id == id, 'track_condition'].item()

print ("Adding horse statistics (index, rank, recent avg and results)...")
for id in id_list:
    df.loc[df['horse_id'] == id, 'horse_index'] = idx
    idx += 1

    breaker = '/'
    recent = []
    for index, record in df[df['horse_id'] == id].iterrows():
        pos = int(record.finishing_position)
        if pos == 1:
            df.loc[index, 'horse_win'] = 1
            df.loc[index, 'horse_rank_top_3'] = 1
            df.loc[index, 'horse_rank_top_50_percent'] = 1
        elif pos <= 3:
            df.loc[index, 'horse_rank_top_3'] = 1
            df.loc[index, 'horse_rank_top_50_percent'] = 1
        else:
            num_horses = max(map(float,df.loc[df['race_id'] == record.race_id, 'finishing_position']))
            if float(pos) / num_horses <= 0.5:
                df.loc[index, 'horse_rank_top_50_percent'] = 1

        if (len(recent)) != 0:
            df.loc[index, 'recent_ave_rank'] = sum(map(float,recent))/float(len(recent))

        df.loc[index, 'recent_6_runs'] = breaker.join(recent)
        recent.append(record.finishing_position)
        if len(recent) > 6:
            recent = recent[1:]

print ("Adding jockey statistics (index, avg rank)...")
idx = max(idx, 20000)
for name in jockey_list:
    df.loc[df.jockey == name, 'jockey_index'] = idx
    idx += 1
    jockey_record = df.loc[df.jockey == name]
    jockey_position = jockey_record.loc[df.race_id <= '2016-327', 'finishing_position']
    if len(jockey_position) != 0:
        df.loc[df.jockey == name, 'jockey_ave_rank'] = \
            sum(map(float, jockey_position))/float(len(jockey_position))

print ("Adding trainer statistics (index, avg rank)...")
idx = max(idx, 40000)
for name in trainer_list:
    df.loc[df.trainer == name, 'trainer_index'] = idx
    idx += 1
    trainer_record = df.loc[df.trainer == name]
    trainer_position = trainer_record.loc[df.race_id <= '2016-327', 'finishing_position']
    if len(trainer_position) != 0:
        df.loc[df.trainer == name, 'trainer_ave_rank'] = \
            sum(map(float, trainer_position))/float(len(trainer_position))


print ("Creating training.csv and testing.csv files...")
df_train = df.loc[df.race_id <= '2016-327']
df_test = df.loc[df.race_id > '2016-327']

df_train.to_csv('data/training.csv')
df_test.to_csv('data/testing.csv')

print ("Complete!")
