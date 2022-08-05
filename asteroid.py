import torch
import os

import numpy as np
import matplotlib
import requests
import torch
import torchaudio

from torchaudio.models.decoder import download_pretrained_files
from torchaudio.models.decoder import ctc_decoder

import aligner as al
import datetime

start = 0
end = 30

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

SPEECH_FILE = "data/spider_man.wav"
# SPEECH_FILE = "data/spider_man_short_vocals.wav"
# SPEECH_FILE = "output/spider_man_short/vocals.wav"


#Load waveform
waveform, sample_rate = torchaudio.load(SPEECH_FILE)

print(waveform.shape)
# waveform = waveform.mean(axis=0).unsqueeze(0)
waveform = waveform[:, start*sample_rate:end*sample_rate]
waveform = waveform.reshape(1, 2, -1)
print(waveform.shape)

# #Resample waveform if needed
# if sample_rate != bundle.sample_rate:
#     waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


print(waveform.shape)
waveform = waveform.to(device)

# loading umxhq four target separator
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')

print(separator.sample_rate, sample_rate)

# waveform = torch.rand((1, 1, 100000))
print(waveform.shape)
estimates = separator(waveform)

print(estimates.shape)

torchaudio.save("test0.wav", estimates[0,0,:,:], sample_rate, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("test1.wav", estimates[0,1,:,:], sample_rate, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("test2.wav", estimates[0,2,:,:], sample_rate, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("test3.wav", estimates[0,3,:,:], sample_rate, encoding="PCM_S", bits_per_sample=16)