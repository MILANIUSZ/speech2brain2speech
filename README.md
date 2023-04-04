## Towards Naturalistic BCI: Applying Deep Learning to Decode Brain Activity During Passive Listening to Speech
Milan Andras Fodor
Thesis work

The study explores the possibility of decoding listened speech and audio from intracranial EEG (iEEG) recordings of the brain using deep learning models. We used the 'Open multimodal iEEG-fMRI dataset' in which participants watched a movie while their iEEG data was recorded. We proposed a sliding window approach to extract high gamma features, which was fed into various deep neural networks (FC-DNN and CNN). The target of the DNNs was the mel-spectrogram of the movie audio. The model was trained, validated, and tested on iEEG data and corresponding mel-spectrograms. Although the synthesised speech using WaveGlow neural vocoder was not intelligible yet, the DNNs could generate audio where the speech and silent segments are separable. This study is unique in that it focuses on synthesising speech that was listened to by the participant rather than spoken by the subject. This approach might take us closer to a BCI system and might help to understand the cognitive aspects of speech.

# *Listened speech decoding from intracarnial signals*

The aim of this work is to reconstruct the speech that was heard by an individual in a real world scenario, with the use of intracarnial data and deep learning methods.


*Dataset:*  https://www.nature.com/articles/s41597-022-01173-0  
download: https://openneuro.org/datasets/ds003688/versions/1.0.7/download  


*Environment:*  Docker - public: thegeeksdiary/tensorflow-jupyter-gpu  
https://thegeeksdiary.com/2023/01/29/how-to-setup-tensorflow-with-gpu-support-using-docker/


Libary list:
