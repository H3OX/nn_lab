import joblib
import pandas as pd
import numpy as np

tess = joblib.load('data/data_tess.pkl')
rvds = joblib.load('data/data_ravdess.pkl')

tess_rvds = tess.append(rvds, ignore_index=True)
joblib.dump(tess_rvds, 'data/tess_rvds.pkl')