import os
import glob
import librosa
import numpy as np
import datetime
from numpy import savez
from hparam import hparam as hp
import pyworld as pw
from data_load import get_mfccs_and_phones
from tqdm import tqdm

TIMIT_TRAIN_WAV = 'TIMIT/TRAIN/*/*/*.WAV'
TIMIT_TEST_WAV = 'TIMIT/TEST/*/*/*.WAV'

train_mfccs = None
train_phns = None
test_mfccs = None
test_phns = None

if __name__ == "__main__":
    dataset_path = "/home/john/datasets"
    preproc_data_path = "./"
    s = datetime.datetime.now()

    train_wav_files = glob.glob(os.path.join(dataset_path, TIMIT_TRAIN_WAV))
    test_wav_files = glob.glob(os.path.join(dataset_path, TIMIT_TEST_WAV))

    print('Starting pre-processing train dataset...')
    print("Extracting and saving features from wav files...")
    for index in tqdm(range(len(train_wav_files))):
        mfcc, phn = get_mfccs_and_phones(train_wav_files[index])
        if index == 0:
            train_mfccs = mfcc
            train_phns = phn
        else:
            train_mfccs = np.concatenate((train_mfccs, mfcc))
            train_phns = np.concatenate((train_phns, phn))


    print('Pre-processing of train dataset has finished!')

    print('Starting pre-processing test dataset...')
    print("Extracting and saving features from wav files...")
    for index in tqdm(range(len(test_wav_files))):
        mfcc, phn = get_mfccs_and_phones(test_wav_files[index])
        if index == 0:
            test_mfccs = mfcc
            test_phns = phn
        else:
            test_mfccs = np.concatenate((test_mfccs, mfcc))
            test_phns = np.concatenate((test_phns, phn))

    np.savez("train.npz", mfccs=train_mfccs, phns=train_phns)
    np.savez("test.npz", mfccs=test_mfccs, phns=test_phns)

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))

