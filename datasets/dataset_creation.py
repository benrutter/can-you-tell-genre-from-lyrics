import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('mxm_dataset.db')
query = "SELECT * FROM lyrics"
lyrics = pd.read_sql(query, conn)

track_ids = []
genres = []
with open('msd_beatunes_map.cls') as file:
    for row in file:
        track_ids.append(row.split()[0])
        try:
            genres.append(row.split()[1])
        except:
            genres.append(np.nan)

genre = pd.DataFrame.from_dict({'track_id':track_ids, 'genre':genres})
genre = genre.iloc[7:]

genre['track_id'] = genre['track_id'].astype(str)
lyrics['track_id'] = lyrics['track_id'].astype(str)

combined = genre.set_index('track_id').join(lyrics.set_index('track_id'))

combined[['genre', 'word', 'count']].to_csv('genre_and_lyrics.csv')
