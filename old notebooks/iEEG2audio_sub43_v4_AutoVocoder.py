'''
iEEG-to-speech synthesis using AutoVocoder
written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
last version: March 9, 2023
'''

import numpy as np
import mne
import pandas as pd
import mne_bids
import matplotlib.pyplot as plt


# for speech processing
import WaveGlow_functions
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
# from imblearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import torch
import datetime
import gc
import os
import pickle
import sys
import argparse
from datetime import datetime, timedelta
from subprocess import call, check_output, run
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
import scipy.io as sio
import skimage.transform
import soundfile as sf
import tensorflow

from utils_iEEG import resample

import keras
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

# '''
# do not use all GPU memory
from keras.backend.tensorflow_backend import set_session
config = tensorflow.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tensorflow.Session(config=config))
# '''

import scipy
import scipy.io
# import scipy.io.wavfile
import scipy.io.wavfile as io_wav


#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr, windowLength=0.05, frameshift=0.01, bandpass_min=70, bandpass_max=170):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    # sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    sos = scipy.signal.iirfilter(4, [bandpass_min/(sr/2),bandpass_max/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    # sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    # data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    # sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    # data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked

# WaveGlow / Tacotron2 / STFT parameters for audio data
# samplingFrequency = 16000
samplingFrequency = 22050
# samplingFrequency_EEG = 2048
winL_EEG = 0.05
# frameshift_EEG = 0.01 # 10 ms
frameshift_EEG = 0.01161 # 11.61 ms, same as AutoVocoder
frameshift_speech = 256 # 11.61 ms, same as AutoVocoder
modelOrder_EEG = 0
# modelOrder_EEG = 1
# modelOrder_EEG = 2
modelOrder_EEG = 4
# modelOrder_EEG = 10
# stepSize_EEG = 1
# stepSize_EEG = 2
stepSize_EEG = 5


stft = WaveGlow_functions.TacotronSTFT(
        filter_length=1024,
        hop_length=frameshift_speech,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=samplingFrequency,
        mel_fmin=0,
        mel_fmax=8000)

# AutoVocoder parameters
n_melspec = 256
# hop_length_iEEG = 220 # 10 ms
# hop_length_AutoVocoder = 256

wavfile = '/media/wd-black-4tb/shared_data/OpenNeuro-iEEG-fMRI/pippi_remix_1_less_scary_loud_wmv8_1_22k.wav'
# mel_data = WaveGlow_functions.get_mel(wavfile, stft)
# mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))

# pippi_remix_1_less_scary_loud_wmv8_1_22k.ae_spec

# load AutoVocoder/melspec
# with open('/media/wd-black-4tb/shared_data/OpenNeuro-iEEG-fMRI/pippi_remix_1_less_scary_loud_wmv8_1_22k.ae_spec', 'rb') as handle:
with open('/media/wd-black-4tb/shared_data/OpenNeuro-iEEG-fMRI/pippi_remix_1_less_scary_loud_wmv8_1_22k.ae_spec_fon4', 'rb') as handle:
    mel_data = pickle.load(handle)

# mel_data_0 = mel_data.copy()
# print('interpolate / resize')
# interpolate_ratio = hop_length_AutoVocoder / hop_length_iEEG
# mel_data = skimage.transform.resize(mel_data, \
                    # (int(mel_data.shape[0] * interpolate_ratio), mel_data.shape[1]), preserve_range=True)

bids_dir = '/media/wd-black-4tb/shared_data/OpenNeuro-iEEG-fMRI/ds003688-download/'
subjects = mne_bids.get_entity_vals(bids_dir, 'subject')
print(subjects)

# subject = '07'
# subject = '01'
subject = '43' # Julia recommended
# n_eeg_channels = 101 # for subject 01
acquisition = 'clinical'
task = 'film'
datatype = 'ieeg'
session = 'iemu'

channels_path = mne_bids.BIDSPath(subject=subject,
                                    session=session,
                                    suffix='channels',
                                    extension='tsv',
                                    datatype=datatype,
                                    task=task,
                                    acquisition=acquisition,
                                    root=bids_dir)

channels = pd.read_csv(str(channels_path.match()[0]), sep='\t', header=0, index_col=None)

# print(channels)

data_path = mne_bids.BIDSPath(subject=subject,
                                    session=session,
                                    suffix='ieeg',
                                    extension='vhdr',
                                    datatype=datatype,
                                    task=task,
                                    acquisition=acquisition,
                                    root=bids_dir)
raw = mne.io.read_raw_brainvision(str(data_path.match()[0]), scale=1.0, preload=False, verbose=True)
raw.set_channel_types({ch_name: str(x).lower()
                if str(x).lower() in ['ecog', 'seeg', 'eeg'] else 'misc'
                                for ch_name, x in zip(raw.ch_names, channels['type'].values)})
raw.drop_channels([raw.ch_names[i] for i, j in enumerate(raw.get_channel_types()) if j == 'misc'])

print(raw.info)

bad_channels = channels['name'][(channels['type'].isin(['ECOG', 'SEEG'])) & (channels['status'] == 'bad')].tolist()


raw.info['bads'].extend([ch for ch in bad_channels])
raw.drop_channels(raw.info['bads'])

n_eeg_channels = int(raw.info['nchan']) # for subject 01
print('n_eeg_channels', n_eeg_channels)
# raise


raw.load_data()

raw.notch_filter(freqs=np.arange(50, 251, 50))

# raw.plot()
# plt.show()

raw_car, _ = mne.set_eeg_reference(raw.copy(), 'average')

# gamma = raw_car.copy().filter(60, 120).apply_hilbert(envelope=True).get_data().T

gamma = raw_car.copy().filter(60, 120).apply_hilbert(envelope=True).get_data()#.T
print('raw_car.shape:', raw_car._data.shape, 'gamma shape: ', gamma.shape)

# plt.subplot(211)
# plt.imshow(raw_car._data[:, 0:2000], aspect='auto')
# plt.subplot(212)
# plt.imshow(np.rot90(gamma[0:2000]), aspect='auto')
# plt.show()
# raise
# output: (798828, 101) # 390.05 sec


custom_mapping = {'Stimulus/music': 2, 'Stimulus/speech': 1,
                  'Stimulus/end task': 5}  # 'Stimulus/task end' in laan
events, event_id = mne.events_from_annotations(raw_car, event_id=custom_mapping,
                                                         use_rounding=False)


# print(events)

# raise



# raw_car.plot(events=events, start=0, duration=30, color='gray', event_color={2: 'g', 1: 'r'}, bgcolor='w')
# plt.show()

# Cut only the stimuli (when watching the movie 6.5 min)

# gammaonlystimuli = gamma[43672:842500, :]

# raw_car_cut = raw_car._data.copy()
# raw_car_cut = gamma
# print(raw_car_cut.shape)




# '''

#get EEG SR
samplingFrequency_EEG=raw_car.info['sfreq']

# +
# Get the start and end times of the data to keep
# start_time = events[0, 0] / raw_car.info['sfreq']
# end_time = events[-1, 0] / raw_car.info['sfreq']

print('before cut: ', raw_car._data.shape, mel_data.shape)

'''
# only first speech part
start_time = events[1, 0] / raw_car.info['sfreq']
end_time = events[2, 0] / raw_car.info['sfreq']
# cut from speech as well
# 30..60 sec, 10ms frame shift -> 100 frames /sec
mel_data = mel_data[30*100 : 60*100]

# Convert start and end times to indices
start_idx, end_idx = raw_car.time_as_index([start_time, end_time])

print('time', start_idx, end_idx)
# raise

# Slice the data to keep only the required part
raw_car_cut = raw_car._data[:, start_idx:end_idx]
# print(raw_car_cut.shape)
'''

raw_car_cut = np.empty((n_eeg_channels,0))
mel_data_cut = np.empty((0,n_melspec))


# for i in range(6):
for i in range(6):
    start_time = events[2*i+1, 0] / raw_car.info['sfreq']
    end_time = events[2*i+2, 0] / raw_car.info['sfreq']
    start_idx, end_idx = raw_car.time_as_index([start_time, end_time])
    print(i, 'iEEG index', start_idx, end_idx, end_idx-start_idx)
    n_frames_per_sec = int(1 / frameshift_EEG)
    print(i, 'melspec index', (2*i+1)*30*n_frames_per_sec, (2*i+2)*30*n_frames_per_sec, (2*i+2)*30*n_frames_per_sec-(2*i+1)*30*n_frames_per_sec)
    # raw_car_cut1 = raw_car._data[:, start_idx:end_idx]
    raw_car_cut1 = gamma[:, start_idx:end_idx]
    raw_car_cut = np.append(raw_car_cut, raw_car_cut1, axis=1)
    mel_data_cut1 = mel_data[(2*i+1)*30*n_frames_per_sec : (2*i+2)*30*n_frames_per_sec]
    mel_data_cut = np.append(mel_data_cut, mel_data_cut1, axis=0)
# raise
mel_data = mel_data_cut

print('after cut: ', raw_car_cut.shape, mel_data.shape)
# raise


gamma_resampled = resample(raw_car_cut.T, int(samplingFrequency / frameshift_speech), int(samplingFrequency_EEG))
print('gamma', raw_car_cut.shape, gamma_resampled.shape)
gamma_resampled = gamma_resampled.T

# plt.imshow(gamma_resampled[:, 0:2000], aspect='auto')
# plt.show()


'''
# for sub01
if subject == '01':
    raw_car_cut = raw_car_cut[:, 43672:842500]
elif subject == '07':
    raw_car_cut = raw_car_cut[:, 43672:842500]
'''



# crop to first 10 seconds
# raw_car_cut = raw_car_cut[:, 0:10*2048]
# mel_data = mel_data[0:10*100]

# gamma sampling rate: 2048 Hz




n_freq_bands = 1

#Extract HG features
# print('calculating Hilbert...', raw_car_cut.shape)
# eeg_fft = np.empty((n_max_frames, n_freq_bands, n_eeg_channels * (2 * modelOrder_EEG + 1) ))
# feat_Hilbert_1 = extractHG(raw_car_cut,samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=1, bandpass_max=200)
# feat_Hilbert_1 = extractHG(np.rot90(raw_car_cut),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=1, bandpass_max=50)
# feat_Hilbert_2 = extractHG(np.rot90(raw_car_cut),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=51, bandpass_max=100)
# feat_Hilbert_3 = extractHG(np.rot90(raw_car_cut),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=101, bandpass_max=150)
# feat_Hilbert_4 = extractHG(np.rot90(raw_car_cut),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=151, bandpass_max=200)

#Stack features
# feat_Hilbert_1 = stackFeatures(feat_Hilbert_1,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
# feat_Hilbert_2 = stackFeatures(feat_Hilbert_2,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
# feat_Hilbert_3 = stackFeatures(feat_Hilbert_3,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
# feat_Hilbert_4 = stackFeatures(feat_Hilbert_4,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)

# print(feat_Hilbert_1.shape)
# raise

print('gamma', gamma_resampled.T.shape, 'mel', mel_data.shape)
feat_gamma = stackFeatures(gamma_resampled.T,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)

feat_Hilbert = np.zeros((len(feat_gamma), n_freq_bands, n_eeg_channels * (2 * modelOrder_EEG + 1)))
print(feat_Hilbert[:, 0, :].shape, feat_gamma.shape)
feat_Hilbert[:, 0, :] = feat_gamma
# feat_Hilbert[:, 0, :] = feat_Hilbert_1
# feat_Hilbert[:, 1, :] = feat_Hilbert_2
# feat_Hilbert[:, 2, :] = feat_Hilbert_3
# feat_Hilbert[:, 3, :] = feat_Hilbert_4

feat_Hilbert = feat_Hilbert.reshape(-1, n_freq_bands * n_eeg_channels * (2 * modelOrder_EEG + 1))

'''
plt.subplot(211)
plt.imshow(np.rot90(feat_Hilbert_1), aspect='auto')

plt.subplot(212)
plt.imshow(np.rot90(mel_data), aspect='auto')
plt.show()
'''
eeg = feat_Hilbert

min_len = np.min((len(eeg), len(mel_data)))
print('min: ', len(eeg), len(mel_data), min_len)
# raise
eeg = eeg[0:min_len]
mel_data = mel_data[0:min_len]

print('mel & iEEG: ', mel_data.shape, feat_Hilbert.shape)

# train-validation-test split
eeg_train = eeg[0 : int(len(eeg) * 0.8)]
eeg_valid = eeg[int(len(eeg) * 0.8) : int(len(eeg) * 0.9)]
eeg_test =  eeg[int(len(eeg) * 0.9) : ]    

melspec_train = mel_data[0 : int(len(mel_data) * 0.8)]
melspec_valid = mel_data[int(len(mel_data) * 0.8) : int(len(mel_data) * 0.9)]
melspec_test =  mel_data[int(len(mel_data) * 0.9) : ]    

# def eeg_transform(x):
    # print(x)
    # return np.exp(x)

# scale input to [0-1]
eeg_scaler = MinMaxScaler()
# eeg_scaler = MinMaxScaler(feature_range=(0, 10))
# eeg_scaler_2 = FunctionTransformer(eeg_transform)
# eeg_scaler_1 = MinMaxScaler(feature_range=(0, 1))
# eeg_scaler = Pipeline(steps=[('scaler', eeg_scaler_1), ('transformer', eeg_scaler_2)], memory='sklearn_tmp_memory')

# eeg_train = np.log(eeg_train)
# eeg_valid = np.log(eeg_valid)
# eeg_test = np.log(eeg_test)

# eeg_scaler = StandardScaler(with_mean=True, with_std=True)
eeg_train_scaled = eeg_scaler.fit_transform(eeg_train)
eeg_valid_scaled = eeg_scaler.transform(eeg_valid)
eeg_test_scaled  = eeg_scaler.transform(eeg_test)

# del raw_eeg
# del eeg

# scale outpit mel-spectrogram data to zero mean, unit variances
melspec_scaler = StandardScaler(with_mean=True, with_std=True)
# melspec_scaler = MinMaxScaler()
melspec_train_scaled = melspec_scaler.fit_transform(melspec_train)
melspec_valid_scaled = melspec_scaler.transform(melspec_valid)
melspec_test_scaled  = melspec_scaler.transform(melspec_test)

# plt.subplot(211)
# plt.imshow(np.rot90(eeg_train[0:1000]), aspect='auto')

# plt.subplot(212)
# plt.imshow(np.rot90(melspec_train[0:1000]), aspect='auto')
# plt.show()

plt.subplot(411)
plt.hist(eeg_train.ravel(), bins=1000)
plt.subplot(412)
plt.hist(eeg_train_scaled.ravel(), bins=1000)
plt.subplot(413)
plt.hist(melspec_train.ravel(), bins=100)
plt.subplot(414)
plt.hist(melspec_train_scaled.ravel(), bins=100)
plt.show()
# raise

plt.subplot(411)
plt.imshow(np.rot90(eeg_train[0:1000]), aspect='auto')
plt.subplot(412)
plt.imshow(np.rot90(eeg_train_scaled[0:1000]), aspect='auto')
plt.subplot(413)
plt.imshow(np.rot90(melspec_train[0:1000]), aspect='auto')
plt.subplot(414)
plt.imshow(np.rot90(melspec_train_scaled[0:1000]), aspect='auto')
plt.show()

'''
plt.subplot(211)
plt.imshow(np.rot90(eeg_train_scaled[0:1000]), aspect='auto')

plt.subplot(212)
plt.imshow(np.rot90(melspec_train_scaled[0:1000]), aspect='auto')
plt.show()
'''


# 5 hidden layers, with 1000 neuron on each layer
model = Sequential()
model.add(
    Dense(
        1000,
        input_dim=n_freq_bands * n_eeg_channels * (2 * modelOrder_EEG + 1),
        kernel_initializer='normal',
        activation='relu'))
model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
model.add(
    Dense(
        n_melspec,
        kernel_initializer='normal',
        activation='linear'))

# compile model
model.compile(
    loss='mean_squared_error',
    metrics=['mean_squared_error'],
    optimizer='adam')
# earlystopper = EarlyStopping(
    # monitor='val_mean_squared_error',
    # min_delta=0.0001,
    # patience=3,
    # verbose=1,
    # mode='auto')

# callbacks from Laci
earlystopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=3, verbose=1, mode='auto')
lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=2, verbose=1, factor=0.5, min_lr=0.0001) 


print(model.summary())

if not (os.path.isdir('models_iEEG_to_melspec/')):
    os.mkdir('models_iEEG_to_melspec/')

# early stopping to avoid over-training
# csv logger
current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(
    date=datetime.now())
print(current_date)
# n_eeg_channels * (2 * modelOrder_EEG + 1)
model_name = 'models_iEEG_to_melspec/iEEG-Hilbert_to_AV_DNN_modelOrder-' + str(modelOrder_EEG).zfill(2) + '_freqBands-1_' + 'sub' + subject + '_' + current_date
logger = CSVLogger(model_name + '.csv', append=True, separator=';')
checkp = ModelCheckpoint(
    model_name +
    '_weights_best.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min')

# save model
model_json = model.to_json()
with open(model_name + '_model.json', "w") as json_file:
    json_file.write(model_json)

# serialize scalers to pickle
# pickle.dump(eeg_scaler, open(model_name + '_eeg_scaler.sav', 'wb'))
# pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))


# plt.subplot(211)
# plt.imshow(np.rot90(eeg_train_scaled[0:1000]), aspect='auto', cmap='gray')
# plt.subplot(212)
# plt.imshow(np.rot90(melspec_train_scaled[0:1000]), aspect='auto', cmap='gray')
# plt.show()

# eeg_train_scaled = eeg_train_scaled[0:3000]
# melspec_train_scaled = melspec_train_scaled[0:3000]

plt.subplot(211)
plt.imshow(np.rot90(eeg_train_scaled), aspect='auto', cmap='gray', vmin=0, vmax=1)
plt.subplot(212)
plt.imshow(np.rot90(melspec_train_scaled), aspect='auto', cmap='gray',vmin=-1, vmax=1)
plt.show()

# Run training
# validation_data=(eeg_valid_scaled, melspec_valid_scaled),
history = model.fit(eeg_train_scaled, melspec_train_scaled,
                    epochs=100, batch_size=128, shuffle=True, verbose=1,
                    callbacks = [earlystopper, lrr, logger, checkp],
                    validation_data=(eeg_valid_scaled, melspec_valid_scaled),
                    )

# here the training of the DNN is finished
# load back best weights
model.load_weights(model_name + '_weights_best.h5')
# remove model file
# os.remove(model_name + '_weights_best.h5')

# temp: use train data for testing...
# eeg_test_scaled = eeg_train_scaled[0:2000]
# melspec_test_scaled = melspec_train_scaled[0:2000]
# melspec_test = melspec_train[0:2000]

# melspec_predicted = model.predict(eeg_test_scaled[0:500])
melspec_predicted = model.predict(eeg_test_scaled)
# melspec_predicted = melspec_predicted[0:500]
# test_melspec = test_melspec[]




# ult_predicted = ult_predicted.reshape(-1, NumVectors, PixPerVector_resized)
# ult_test = ult_test.reshape(-1, NumVectors, PixPerVector_resized)

plt.subplot(311)
plt.imshow(np.rot90(eeg_test_scaled[0:2000]), aspect='auto')

plt.subplot(312)
plt.imshow(np.rot90(melspec_test_scaled[0:2000]), aspect='auto')

plt.subplot(313)
plt.imshow(np.rot90(melspec_predicted[0:2000]), aspect='auto')

plt.savefig(model_name + '_AV_scaled.png')
# plt.show()
plt.close()


# AutoVocoder requirements
import json
import torch
from scipy.io.wavfile import write
from complexdataset import ComplexDataset
from torch.utils.data import DataLoader
from models import Generator
from utils import load_checkpoint
from env import AttrDict

# load AutoVocoder
with open('/home/csapot/deep_learning_ultrasound/2022_2023_2/AutoVocoder/cp_autovocoder_CsTG/config.json') as f:
    data = f.read()
global h
json_config = json.loads(data)
h = AttrDict(json_config)
torch.manual_seed(h.seed)
global device
if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

generator = Generator(h).to(device)
state_dict_g = load_checkpoint('/home/csapot/deep_learning_ultrasound/2022_2023_2/AutoVocoder/cp_autovocoder_CsTG/g_00060000', device)
generator.load_state_dict(state_dict_g['generator'])

generator = generator.float()

melspec_predicted = melspec_scaler.inverse_transform(melspec_predicted)

dir_synth = 'models_iEEG_to_melspec/'
basefile = model_name

# interpolate back to original frame shift: 256 samples (vs 270)
# print('interpolate / resize')
# interpolate_ratio = hop_length_iEEG / hop_length_AutoVocoder
# melspec_test = skimage.transform.resize(melspec_test, \
    # (int(melspec_test.shape[0] * interpolate_ratio), melspec_test.shape[1]), preserve_range=True)
# melspec_predicted = skimage.transform.resize(melspec_predicted, \
    # (int(melspec_predicted.shape[0] * interpolate_ratio), melspec_predicted.shape[1]), preserve_range=True)

plt.figure(figsize=(10,6))
plt.subplot(211)
plt.imshow(np.rot90(melspec_test), cmap='gray')
plt.title('original')
plt.subplot(212)
plt.imshow(np.rot90(melspec_predicted), cmap='gray')
plt.title('predicted')
plt.savefig(dir_synth + os.path.basename(basefile) + '_AV-melspec-predicted-n' + str(n_melspec) + '.png')
plt.close()

# synthesis of original file // interpolated
mel_data_gpu = torch.from_numpy(np.expand_dims(melspec_test, axis=0)).to(device).float()
y_g_hat = generator(mel_data_gpu)
y_g_hat_fig = y_g_hat.cpu().detach().numpy().squeeze()
output_filename = dir_synth + os.path.basename(basefile) + '_AV_synth_orig.wav'
write(output_filename, h.sampling_rate, y_g_hat_fig)

# synthesis of predicted file
melspec_predicted_gpu = torch.from_numpy(np.expand_dims(melspec_predicted, axis=0)).to(device).float()
y_g_hat_predicted = generator(melspec_predicted_gpu)
y_g_hat_predicted_fig = y_g_hat_predicted.cpu().detach().numpy().squeeze()
output_filename = dir_synth + os.path.basename(basefile) + '_AV_synth_pred.wav'
write(output_filename, h.sampling_rate, y_g_hat_predicted_fig)

print('synthesis done: ', basefile)
