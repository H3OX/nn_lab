import pandas as pd
import joblib

tess_rvds = joblib.load('../data/tess_rvds.pkl')
savee = joblib.load('../data/data_savee.pkl')
tess_rvds_savee = tess_rvds.append(savee, ignore_index=True)
joblib.dump(tess_rvds_savee, '../data/tess_rvds_savee.pkl')