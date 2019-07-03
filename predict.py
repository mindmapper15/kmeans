from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
from glob import glob
from data_load import read_mfccs_and_phones
import matplotlib.pyplot as plt
from tqdm import tqdm

data_test_path = "./TIMIT/TEST/DR1/FAKS0/*.npz"
npz_test_files = glob(data_test_path)

mfcc_test_list = None
phns_test_list = None

for i in tqdm(range(len(npz_test_files))):

    if mfcc_test_list is None:
        mfcc_test_list, phns_test_list = read_mfccs_and_phones(npz_test_files[i])
    else:
        mfccs, phns = read_mfccs_and_phones(npz_test_files[i])
        mfcc_test_list = np.concatenate((mfcc_test_list, mfccs))
        phns_test_list = np.concatenate((phns_test_list, phns))

phns_test_list = phns_test_list.astype(np.int32)

filename = '
model = joblib.load('kmeans_model.sav')
predicts = model.predict(mfcc_test_list)

mfccs_per_pronuns = [ [] for _ in range(61) ]
pred_per_pronuns = [ [] for _ in range(61) ]

mfccs_per_pronun = []
pred_per_pronun = []
phn = -1
for i in range(len(phns_test_list)):
    if phn != -1 and phn != phns_test_list[i]:
        mfccs_per_pronuns[phn].append(mfccs_per_pronun)
        pred_per_pronuns[phn].append(pred_per_pronun)
        
        mfccs_per_pronun = []
        pred_per_pronun = []
        
    phn = phns_list[i]
    mfccs_per_pronun.append(mfcc_test_list[i])
    pred_per_pronun.append(predicts[i])

# For last phns
mfccs_per_pronuns[phn].append(mfccs_per_pronun)
pred_per_pronuns[phn].append(pred_per_pronun)


zip_pred_per_pronuns = []
for pred_per_pronun in pred_per_pronuns:
    zip_pred_per_pronun = []
    
    for one_pronun in pred_per_pronun:
        seq_pronun = []
        current_pred = -1
        for predict in one_pronun:
            
            if current_pred != -1 and current_pred != predict:
                seq_pronun.append(current_pred)
            current_pred = predict
        seq_pronun.append(current_pred)
        zip_pred_per_pronun.append(seq_pronun)
    zip_pred_per_pronuns.append(zip_pred_per_pronun)
    
for pred_per_pronun in pred_per_pronuns[9]:
    print(pred_per_pronun)

print()
for zip_pred_per_pronun in zip_pred_per_pronuns[9]:
    print(zip_pred_per_pronun)
    
## predict mfccs
## zip seq
## show or print distibution of seq


