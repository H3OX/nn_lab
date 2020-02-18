import pandas as pd
import joblib
import numpy as np
import librosa as lr
import os
import time
from sklearn.preprocessing import LabelEncoder

path = 'TESS'
encoder = LabelEncoder()
data_tess = []

start = time.time()
print('Writing files...')
for subdir, dirs, files in os.walk(path):
    for file in files:
        if file == '.DS_Store':
            pass
        else:
            target = file.split('_')[2].split('.')[0]
            y, sr = lr.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(lr.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
            sample = mfccs, target
            data_tess.append(sample)
end = time.time()

print(f'Took {end - start} seconds to write')

X, y = zip(*data_tess)

dataset_tess = pd.DataFrame(X, y)
dataset_tess = dataset_tess.reset_index()
dataset_tess.rename(columns={
    'index': 'target'
}, inplace=True)
dataset_tess['target'].replace({'ps': 'surprised', 'fear': 'fearful'}, inplace=True)
dataset_tess.replace({
    'angry': 0,
    'disgust': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}, inplace=True)


