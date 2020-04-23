# -*- coding: utf-8 -*-
import os
import time

import joblib
import librosa as lr
import numpy as np
import pandas as pd


emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

start = time.time()

data = []

path = '../data/RAVDESS'

print('Writing files...')
for subdir, dirs, files in os.walk(path):

    for file in files:
        if file == '.DS_Store':
            pass
        else:
            target = emotions[str(file.split('-')[2])]
            y, sr = lr.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(lr.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
            sample = mfccs, target
            data.append(sample)

end = time.time()
print(f'Writing finished in {end - start} seconds')

X, y = zip(*data)

dataset = pd.DataFrame(X, y)

dataset = dataset.reset_index()
dataset.rename(columns={
    'index': 'target'
}, inplace=True)
dataset = dataset[dataset['target'] != 'calm']
dataset['target'].replace({
    'angry': 0,
    'disgust': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}, inplace=True)

joblib.dump(dataset, '../data/r.pkl')


# Dataset ready

# model = nn.Sequential(
#   nn.Linear(30, 60),
#  nn.ReLU(),
# nn.Linear(60, 120),
# nn.ReLU(),
# nn.Linear(120, 8),
# nn.Softmax()
# )
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)

# device = torch.device('cuda:0')
# X_train = torch.tensor(X_train).float().to(device)
# X_test = torch.tensor(X_test).float().to(device)
# y_train = torch.tensor(y_train).to(device)
# y_test = torch.tensor(y_test).to(device)
