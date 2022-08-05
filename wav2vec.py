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
from audio_slicer import AudioSlicer
from source_separater import SourceSeparator
import copy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()

start = args.start
end = args.end


# separator.separate_to_file("data/spider_man_short.wav", "output/spider_man_short")

files = download_pretrained_files("librispeech-4-gram")



matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SPEECH_FILE = "data/spider_man.wav"
# SPEECH_FILE = "data/spider_man_short_vocals_bak.wav"
# SPEECH_FILE = "output/spider_man_short/vocals.wav"

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

print(waveform.shape)
# waveform = waveform.mean(axis=0).unsqueeze(0)
waveform = waveform[:, start*sample_rate:end*sample_rate]
print(waveform.shape)

# #Resample waveform if needed
# if sample_rate != bundle.sample_rate:
#     waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

print(waveform.shape)
waveform = waveform.to(device)

import decoder as dec
import subtitles as sub

subs = sub.Subtitles("data/spider_man_source.srt", start=start, end=end)

decoder = dec.Decoder(waveform,\
    original_sample_rate=sample_rate,\
    model=model,\
    labels=bundle.get_labels(),\
    lexicon_filepath="lexicon.txt",\
    tokens_filepath=files.tokens,\
    lm_filepath="ngram-lm/arpa/spiderman.6gram.arpa",\
    device=device,\
    subtitles=subs,\
    resample_rate=bundle.sample_rate
)

aligner = al.Aligner(subs)

slicer = AudioSlicer()

source_separator = SourceSeparator()

decoded_tokens = []
decoded_timesteps = []

decoded_subtitles = []
decoded_times = []

predicted_subtitles = []
predicted_times = []

previous_tokens = []
previous_timesteps = []

timesteps, slices = slicer.slice(waveform, sample_rate, max_step=3, buffer_region=0.0)

for timestep, audio_slice in zip(timesteps, slices):
    print(timestep)
    # print(waveform.shape)
    # audio_slice = source_separator.separate(audio_slice, sample_rate)
    # print(waveform.shape)
    tokens, timesteps = decoder.decode(timestep, audio_slice)
    tokens = copy.deepcopy(tokens)
    timesteps = copy.deepcopy(timesteps)

    # previous_tokens = tokens
    # previous_timesteps = timesteps

    if len(tokens) > 0:
        decoded_tokens.append(tokens)
        decoded_timesteps.append(timesteps)

        decoded_subtitles.append("".join(tokens))
        decoded_times.append([timesteps[0], timesteps[-1]])

        print(start, timesteps[0])
        
        # subs = sub.Subtitles("data/spider_man_source.srt", start=start+timesteps[0]-5, end=start+timesteps[1]+10)
        aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, previous_tokens=previous_tokens, previous_timesteps=previous_timesteps, subtitles=subs)
        previous_tokens = tokens
        previous_timesteps = timesteps

        # if len(timesteps) > 0:
        #     print("PRE SUB: {}".format("".join(tokens)))
        #     print("PRE STEPS: {}".format([timesteps[0], timesteps[-1]]))
        #     print("POST STEPS: {}".format(aligned_times))

        for subtitle, time in zip(aligned_subtitles, aligned_times):
            predicted_subtitles.append(subtitle)
            predicted_times.append(time)
        
        # print("POST SUB: {}".format(subtitle))
        # print("POST STEPS: {}".format(time))

# print(decoded_tokens)

# for tokens, timesteps in zip(decoded_tokens, decoded_timesteps):
#     aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, previous_tokens=previous_tokens, previous_timesteps=previous_timesteps)

#     if len(timesteps) > 0:
#         print("PRE SUB: {}".format("".join(tokens)))
#         print("PRE STEPS: {}".format([timesteps[0], timesteps[-1]]))
#         print("POST STEPS: {}".format(aligned_times))

#     for subtitle, time in zip(aligned_subtitles, aligned_times):
#         predicted_subtitles.append(subtitle)
#         predicted_times.append(time)
        
#         print("POST SUB: {}".format(subtitle))
#         print("POST STEPS: {}".format(time))
    
#     print("\n")

# for subtitle, time in zip(predicted_subtitles, predicted_times):
#     print("{} {}".format(time, subtitle))

# print(len(predicted_subtitles), len(predicted_times))

sub.write_subtitles("output_subtitles/decoder_out.srt", decoded_subtitles, decoded_times, start_offset=start)
sub.write_subtitles("output_subtitles/aligned_out.srt", predicted_subtitles, predicted_times, start_offset=start)

# predicted_subtitles = []
# predicted_times = []



# for tokens, timesteps in decoder.decode():
#     aligned_subtitles, aligned_times = aligner.align(tokens, timesteps)
#     for subtitle, time in zip(aligned_subtitles, aligned_times):
#         predicted_subtitles.append(subtitle)
#         predicted_times.append(time)
    
# with open("data/spider_man_short.srt", "w") as file:
#     for i, subtitle_times in enumerate(zip(predicted_subtitles, predicted_times)):
#         subtitle, times = subtitle_times
#         file.write(f"{i+1}\n")

#         start_time = str(datetime.timedelta(hours=times[0] / 3600.0 + 0.000000001 + start / 3600.0))[:-3]
#         end_time = str(datetime.timedelta(hours=times[1] / 3600.0 + 0.000000001 + start / 3600.0))[:-3]

#         file.write(start_time + " --> " + end_time + "\n")

#         file.write(subtitle + "\n")
#         file.write("\n")

