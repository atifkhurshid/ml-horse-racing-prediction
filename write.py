def write_csv(a, b, c, path, name):
    columns = ['RaceID','HorseID','HorseWin','HorseRankTop3','HorseRankTop50Percent']
    dataframe = pd.DataFrame({'RaceID':race_id, 'HorseID':horse_id, 'HorseWin':a, 'HorseRankTop3':b, 'HorseRankTop50Percent':c})
    dataframe.to_csv("{0}/{1}_predictions.csv".format(path, name), index=False, sep=',', columns=columns)
