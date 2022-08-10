import os

import numpy as np
import matplotlib
import requests
import torch
import torchaudio
import matplotlib.pyplot as plt

from torchaudio.models.decoder import download_pretrained_files
from torchaudio.models.decoder import ctc_decoder

import copy
import argparse

import asr.aligner as al
from asr.audio_slicer import AudioSlicer
import asr.decoder as dec
import asr.subtitles as sub
from asr.locator import Locator

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()

start = args.start
end = args.end

subs = sub.Subtitles("data/spider_man_source.srt", start=start, end=end)
locator = Locator(start_index=start, end_index=end, min_window=10)


start = subs.subtitles[0].start
end = subs.subtitles[-1].end
start = start.hours*3600.0 + start.minutes*60.0 + start.seconds
start = 3 * (start / 3)
end = end.hours*3600.0 + end.minutes*60.0 + end.seconds

subs = sub.Subtitles("data/spider_man_source.srt")
locator = Locator(start_index=0, end_index=2117, min_window=10)

print(start, end)

files = download_pretrained_files("librispeech-4-gram")



matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SPEECH_FILE = "data/spider_man.wav"
# SPEECH_FILE = "data/spider_man_short_vocals_bak.wav"

if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)

#Load bundle
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
# bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE

#Load model
model = bundle.get_model().to(device)

#Load waveform
waveform, sample_rate = torchaudio.load(SPEECH_FILE)

#Clip between start and end
waveform = waveform[:, int(start*sample_rate):int(end*sample_rate)]
waveform = waveform.to(device)



decoder = dec.Decoder(waveform,\
    original_sample_rate=sample_rate,\
    model=model,\
    labels=bundle.get_labels(),\
    lexicon_filepath="lexicon.txt",\
    tokens_filepath=files.tokens,\
    lm_filepath="ngram-lm/arpa/spiderman.3gram.arpa",\
    device=device,\
    subtitles=subs,\
    resample_rate=bundle.sample_rate
)

aligner = al.Aligner(subs)

slicer = AudioSlicer()

decoded_tokens = []
decoded_timesteps = []

decoded_subtitles = []
decoded_times = []

predicted_subtitles = []
predicted_times = []

previous_tokens = []
previous_timesteps = []

timesteps, slices = slicer.slice(waveform, sample_rate, max_step=3, buffer_region=0.0)

aligned_indices = [0]
aligned_time_list = [0]
actual_indices = [0]
actual_times = [0]
predicted_indices = [0]
predicted_time_list = [0]

for timestep, audio_slice in zip(timesteps, slices):
    print(timestep)
    tokens, timesteps = decoder.decode(timestep, audio_slice)
    tokens = copy.deepcopy(tokens)
    timesteps = copy.deepcopy(timesteps)

    if len(tokens) > 0:
        decoded_tokens.append(tokens)
        decoded_timesteps.append(timesteps)

        decoded_subtitles.append("".join(tokens))
        decoded_times.append([timesteps[0], timesteps[-1]])

        # print(start, timesteps[0])
        

        # print("SEARCH: ", locator.search_range())
        aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, previous_tokens=previous_tokens, previous_timesteps=previous_timesteps, subtitles=subs)
        
        if len(aligned_subtitles) > 0:
            locator.next_index(subs.find(aligned_subtitles[0]))
            
        for subtitle, time in zip(aligned_subtitles, aligned_times):
            # locator.next_index(subs.find(subtitle))
            aligned_indices.append(subs.find(subtitle))
            aligned_time_list.append(time[0])

            actual_subs = subs.subtitles.slice(starts_after={'seconds': time[0]-3}, ends_before={'seconds': time[1]+3})
            # print(time[0]-3, time[1]+3, actual_subs)
            for subt in actual_subs:
                actual_indices.append(subs.find(subt.text))
                actual_times.append(subt.start.hours*3600.0 + subt.start.minutes*60.0 + subt.start.seconds)
        
        if locator.search_range() is not None:
            search_subs = sub.Subtitles("data/spider_man_source.srt", start=locator.search_range()["start"], end=locator.search_range()["end"])
            aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, previous_tokens=previous_tokens, previous_timesteps=previous_timesteps, subtitles=search_subs)

            for subtitle, time in zip(aligned_subtitles, aligned_times):
                predicted_subtitles.append(subtitle)
                predicted_times.append(time)
                predicted_indices.append(subs.find(subtitle))
                predicted_time_list.append(time[0])

#Join sequential duplicates
predicted_subtitles_no_duplicates = [predicted_subtitles[0]]
predicted_times_no_duplicates = [predicted_times[0]]

idx = 1
for i in range(1, len(predicted_subtitles)):
    if predicted_subtitles[i-1] == predicted_subtitles[i]:
        predicted_times_no_duplicates[i-idx][1] = predicted_times[i][1]
        idx += 1
    else:
        predicted_subtitles_no_duplicates.append(predicted_subtitles[i])
        predicted_times_no_duplicates.append(predicted_times[i])

predicted_subtitles = predicted_subtitles_no_duplicates
predicted_times = predicted_times_no_duplicates

plt.plot(aligned_time_list, aligned_indices, 'r.')
plt.plot(actual_times, actual_indices, 'g.')
plt.plot(predicted_time_list, predicted_indices, 'b.')
plt.savefig('plot.png')

sub.write_subtitles("output_subtitles/decoder_out.srt", decoded_subtitles, decoded_times, start_offset=start)
sub.write_subtitles("output_subtitles/aligned_out.srt", predicted_subtitles, predicted_times, start_offset=start)

