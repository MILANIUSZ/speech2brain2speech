import os

import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
import soundfile as sf


import librosa
import librosa.display

from pynwb import NWBHDF5IO
import MelFilterBank as mel


def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram



# +
#audio sampling rate +esampling the audio to a standard sampling rate of 22050 Hz and converting the audio data to mono etc
#audio_path = 'data/stimuli/6min.wav'
#y, sra = librosa.load(audio_path)


# -

'''filename = 'data/stimuli/6min.wav'
audio, srb = sf.read(filename, dtype='float32')'''


audio.shape


# define the output directory path
path_output = r'./features'

# +
audio_sr = 48000
original_audio_sr = 48000
        
#Process Audio
target_SR = 2048
audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
audio_sr = target_SR
scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
os.makedirs(os.path.join(path_output), exist_ok=True)

# -

#othermethods
from scipy import signal
audio, sr = librosa.load('data/stimuli/6min.wav', sr=48000)
audio_resampled = signal.resample(audio, int(len(audio) / sr * 2048))

file_name = 'resampled.wav'
scipy.io.wavfile.write(os.path.join(path_output, file_name), audio_sr, audio_resampled)

winL = 0.05
frameshift = 0.01
melSpec = extractMelSpecs(audio_resampled, 2048, windowLength=winL, frameshift=frameshift)

#import eeg (gamma from preprocessing)
gamma = np.load("gamma.npy")

if melSpec.shape[0] != gamma.shape[0]:
    n_frames = min(melSpec.shape[0], gamma.shape[0])
    melSpec = melSpec[:n_frames, :]
    gamma = gamma[:n_frames, :]

import matplotlib.pyplot as plt
plt.plot(gamma[:,0])
plt.show()

plt.plot(melSpec[:,0])
plt.show()

# +
###resample the eeg isntead
# -

gamma.shape


winL = 0.05
frameshift = 0.01
melSpec48 = extractMelSpecs(audio, 48000, windowLength=winL, frameshift=frameshift)

# +
import resampy
original_sf = 2048.0
target_sf = 48000.0

# Resample the EEG data using resampy
gamma_resampled = resampy.resample(gamma, original_sf, target_sf, axis=0)
# -

gamma_resampled.shape

if melSpec48.shape[0] != gamma_resampled.shape[0]:
    n_frames = min(melSpec48.shape[0], gamma_resampled.shape[0])
    melSpec48 = melSpec48[:n_frames, :]
    gamma_resampled = gamma_resampled[:n_frames, :]

gamma_resampled.shape

melSpec48.shape

# +
import matplotlib.pyplot as plt

# Define time window to plot
start_time = 0  # start time in seconds
end_time = 1  # end time in seconds

# Plot EEG and mel spectrogram for the specified time window
fig, axs = plt.subplots(2, 1, figsize=(20, 10))
axs[0].plot(gamma_resampled[int(start_time*48000):int(end_time*48000), :])
axs[0].set_title('EEG')
axs[1].imshow(melSpec48[int(start_time*48000*0.01):int(end_time*48000*0.01), :].T, origin='lower', aspect='auto', cmap='jet', interpolation='none')
axs[1].set_title('Mel Spectrogram')
plt.show()

# -






