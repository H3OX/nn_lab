import pandas as pd
import joblib
import numpy as np
import librosa as lr
import os
import time

path = '../data/ALL'

data_savee = []

print('writing files...')
start = time.time()
for i in os.listdir(path):
    target = ''
    if i[-8:-6] == '_a':
        target = 'angry'
    elif i[-8:-6] == '_d':
        target = 'disgust'
    elif i[-8:-6] == '_f':
        target = 'fearful'
    elif i[-8:-6] == '_h':
        target = 'happy'
    elif i[-8:-6] == '_n':
        target = 'neutral'
    elif i[-8:-6] == 'sa':
        target = 'sad'
    elif i[-8:-6] == 'su':
        target = 'surprised'

    y, sr = lr.load(os.path.join(path, i), res_type='kaiser_fast')
    mfccs = np.mean(lr.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
    sample = mfccs, target
    data_savee.append(sample)
end = time.time()
print(f'Writing finished in {end - start} seconds')

X, y = zip(*data_savee)
ds_savee = pd.DataFrame(X, y)
ds_savee = ds_savee.reset_index()
ds_savee.rename(columns={
    'index': 'target'
}, inplace=True)
ds_savee['target'].replace({
    'angry': 0,
    'disgust': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}, inplace=True)
joblib.dump(ds_savee, '../data/s.pkl')
