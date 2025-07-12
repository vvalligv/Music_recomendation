import pandas as pd
import numpy as np

num_users = 100
num_songs = 500
num_interactions = 1000

# Randomly generate user-song interactions
users = np.random.randint(0, num_users, num_interactions)
songs = np.random.randint(0, num_songs, num_interactions)
ratings = np.random.randint(1, 6, num_interactions)  # ratings from 1 to 5

df = pd.DataFrame({
    'user_id': users,
    'song_id': songs,
    'rating': ratings
})

print(df.head())
# Save DataFrame to CSV
df.to_csv('music_rec_dataset.csv', index=False)
