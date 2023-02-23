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


# +
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


# -

#audio
audio_path = 'data/stimuli/6min.wav'
ogsr = librosa.get_samplerate(audio_path)
y, sr = librosa.load(audio_path)


y



filename = 'data/stimuli/6min.wav'
audio, sr = sf.read(filename, dtype='float32')
       

audio



 #cant find SR...
        audio_sr = 48000
        original_audio_sr = 48000
        
        #Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
        audio_sr = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        os.makedirs(os.path.join(path_output), exist_ok=True)
        scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sr,scaled) 

#Extract spectrogram
melSpec = extractMelSpecs(scaled,audio_sr,windowLength=winL,frameshift=frameshift)

# +
#feat
#import features (gamma
# -

#Align to EEG features
       # melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
        #adjust length (differences might occur due to rounding in the number of windows)
        #if melSpec.shape[0]!=feat.shape[0]:
          #  tLen = np.min([melSpec.shape[0],feat.shape[0]])
          #  melSpec = melSpec[:tLen,:]
          #  feat = feat[:tLen,:]
        


        #Save everything

