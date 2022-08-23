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
from asr.emission import IndexAligner
import asr.needleman_wunsch as nw
from asr.NWAligner import NWAligner

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()

start = args.start
end = args.end

sstart = args.start
send = args.end

subs = sub.Subtitles("data/spider_man_source.srt", start=start, end=end)
locator = Locator(start_index=start, end_index=end, min_window=10)


start = subs.subtitles[0].start
end = subs.subtitles[-1].end
start = start.hours*3600.0 + start.minutes*60.0 + start.seconds
start = 3 * (start / 3)
end = end.hours*3600.0 + end.minutes*60.0 + end.seconds

subs = sub.Subtitles("data/spider_man_source.srt", start=0, end=2117)
locator = Locator(start_index=0, end_index=2117, min_window=10)

# subs = sub.Subtitles("data/spider_man_source.srt", start=sstart, end=send+100)
# locator = Locator(start_index=sstart, end_index=send+100, min_window=10)

print(start, end)

files = download_pretrained_files("librispeech-4-gram")



matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SPEECH_FILE = "data/spider_man.wav"
# SPEECH_FILE = "data/vocals.wav"
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
    lm_filepath="ngram-lm/arpa/spiderman.6gram.arpa",\
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

index_aligner = IndexAligner(subs)
# subs = sub.Subtitles("data/spider_man_source.srt", start=0, end=10)
nw_aligner = NWAligner(subs)

all_tokens = []
all_timesteps = []

last_best_index = 0

for timestep, audio_slice in zip(timesteps, slices):
    print(timestep)
    tokens, timesteps = decoder.decode(timestep, audio_slice)

    tokens = copy.deepcopy(tokens)
    timesteps = copy.deepcopy(timesteps)

    


    if len(tokens) > 0:
        all_tokens += tokens + ["|"]
        all_timesteps += timesteps + [timesteps[-1]]
    #     all_tokens += ["|"] + tokens
    #     split_idx = [i for i, s in enumerate(tokens) if s == "|"]

    #     for t in np.split(timesteps, split_idx):
    #         all_timesteps.append([t[0], t[-1]])

    # if len(tokens) > 0:
        decoded_tokens.append(tokens)
        decoded_timesteps.append(timesteps)

        decoded_subtitles.append("".join(tokens))
        decoded_times.append([timesteps[0], timesteps[-1]])

    #     # print(start, timesteps[0])
        

        # print("SEARCH: ", locator.search_range())
        # aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, subtitles=subs)
        
        # if len(aligned_subtitles) > 0:
        #     index = locator.next_index(subs.find(aligned_subtitles[0]), tokens, timesteps)
        #     search_range, words, word_times = locator.search_range()
        #     print("Guessed Index: {}".format(subs.find(aligned_subtitles[0])))
        #     if len(nw_aligner.predicted_words) > 10:
        #         search_subs = sub.Subtitles("data/spider_man_source.srt", start=search_range['start'], end=search_range['end'])
        #         nw_aligner = NWAligner(search_subs, words, word_times)





        #     if last_best_index < 100 * (index // 100):
        #         last_best_index = 100 * (index // 100)
        #         search_subs = sub.Subtitles("data/spider_man_source.srt", start=last_best_index, end=last_best_index + 150)
        #         nw_aligner = NWAligner(search_subs)
        #         print("Changing")
        #     print("IN: ", index, 100 * (index // 100))
            
            
    #     for subtitle, time in zip(aligned_subtitles, aligned_times):
    #         # locator.next_index(subs.find(subtitle))
    #         aligned_indices.append(subs.find(subtitle))
    #         aligned_time_list.append(time[0])

    #         actual_subs = subs.subtitles.slice(starts_after={'seconds': time[0]-3}, ends_before={'seconds': time[1]+3})
    #         # print(time[0]-3, time[1]+3, actual_subs)
    #         for subt in actual_subs:
    #             actual_indices.append(subs.find(subt.text))
    #             actual_times.append(subt.start.hours*3600.0 + subt.start.minutes*60.0 + subt.start.seconds)
        

        # if locator.search_range() is not None:
        # # if True:
        #     search_subs = sub.Subtitles("data/spider_man_source.srt", start=locator.search_range()["start"], end=locator.search_range()["end"])
        #     aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, subtitles=search_subs)
        #     # idxs = []
        #     # for i, s in enumerate(subs.subtitles):
        #     #     if s.start.hours*3600.0 + s.start.minutes*60.0 + s.start.seconds > time[0]:
        #     #         idxs.append(i)

        #     #         break
            
        #     # if len(idxs) > 0:
        #     #     search_subs = sub.Subtitles("data/spider_man_source.srt", start=max(0, idxs[0] - 5), end=min(len(subs.subtitles), idxs[-1] + 5))
        #     #     aligned_subtitles, aligned_times = aligner.align(tokens, timesteps, previous_tokens=previous_tokens, previous_timesteps=previous_timesteps, subtitles=search_subs)

        #     if len(aligned_subtitles) > 0:
        #         found_idx = subs.find(aligned_subtitles[0])
        #     else:
        #         found_idx = -1
        
        #     # print("Final idx: {}".format(found_idx))
        #     for subtitle, time in zip(aligned_subtitles, aligned_times):
        #         predicted_subtitles.append(subtitle)
        #         predicted_times.append(time)
        #         predicted_indices.append(subs.find(subtitle))
        #         predicted_time_list.append(time[0])
        # aligned_subtitles, aligned_times = index_aligner.transform(tokens, timesteps)
        # if len(nw_aligner.predicted_words) > 10:
        #     nw_aligner = NWAligner(subs, nw_aligner.predicted_words[-10:], nw_aligner.predicted_word_times[-10:])
        
        aligned_subtitles, aligned_times = nw_aligner.transform(tokens, timesteps)
        good = locator.next_index(nw_aligner.predicted_words_sentence_idx[0])
            # print("Final idx: {}".format(found_idx))
        if good:
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

# plt.plot(aligned_time_list, aligned_indices, 'r.')
# plt.plot(actual_times, actual_indices, 'g.')
# plt.plot(predicted_time_list, predicted_indices, 'b.')
# plt.savefig('plot.png')

sub.write_subtitles("output_subtitles/decoder_out.srt", decoded_subtitles, decoded_times, start_offset=start)
sub.write_subtitles("output_subtitles/aligned_out.srt", predicted_subtitles, predicted_times, start_offset=start)


# [print(x,y) for x,y in zip(all_tokens, all_timesteps)]
nw_aligner = NWAligner(subs)
aligned_subtitles, aligned_times = nw_aligner.transform(all_tokens, all_timesteps)
sub.write_subtitles("output_subtitles/all_out.srt", aligned_subtitles, aligned_times, start_offset=start)

# from collections import defaultdict
# import asr.smith_waterman as sw

# print(all_tokens)

# # num_words = len("".join(tokens).replace("|", " ").split())
# # print(num_words)

# # all_tokens += ["|"] + tokens
# # all_tokens += ["|"] + tokens

# all_tokens = "".join(all_tokens).replace("|", " ").split()
# og_words = subs.whole_text.split()

# # all_tokens += sub.Subtitles("data/spider_man_source.srt", start=send+10, end=send+20).whole_text.split()

# # print(all_tokens)

# whole_text_hash = defaultdict(lambda:-1)
# for i, word in enumerate(og_words[::-1] + all_tokens[::-1]):
#     if word not in whole_text_hash:
#         whole_text_hash[word] = i


# # for i, word in enumerate(all_tokens[::-1]):
# #     if word not in whole_text_hash:
# #         whole_text_hash[word] = i

# whole_text_hash_inv = defaultdict(lambda:-1)

# for k, v in whole_text_hash.items():
#     whole_text_hash_inv[v] = k

# all_tokens_hash = [whole_text_hash[word] for word in all_tokens]
# og_words_hash = [whole_text_hash[word] for word in og_words]

# # if len(all_tokens_hash) < len(og_words_hash):
# #     all_tokens_hash += [0]*abs(len(all_tokens_hash) - len(og_words_hash))


# # x, y = nw.nw(all_tokens_hash, og_words_hash, gap=0, mismatch=0, match=1)
# x, y = nw.nw(all_tokens_hash, og_words_hash, gap=1, mismatch=1, match=1)
# x = [i if i != -1 else None for i in x]
# y = [i if i != -1 else None for i in y]
# # x, y = sw.smith_waterman(all_tokens_hash, og_words_hash)

# x = [whole_text_hash_inv[i] if i is not None else "-" for i in x]
# y = [whole_text_hash_inv[i] if i is not None else "-" for i in y]


# with open("output_subtitles/text.txt", 'w') as f:
#     f.write("")

# with open("output_subtitles/text.txt", 'a') as f:
#     for i,j in zip(x,y):
#         f.write("{}\t{}\n".format(i, j))

# add_t = []
# sent = []

# idx = 0
# idy = 0
# for i, word in enumerate(x):
#     if i-idx >= len(all_timesteps):
#         break

#     if x[i] != "-":
#         add_t.append(all_timesteps[i-idx])
#         sent.append(subs.word_idx_to_sub(idy))
#         # sent.append(idy)
#     if x[i] == "-":
#         idx += 1
#     if  y[i] != "-":
#         idy += 1


# t_no_dupl = [add_t[0]]
# s_no_dupl = [sent[0]]

# idx = 1

# for i in range(1, len(sent)):
#     if sent[i] != s_no_dupl[i-idx]:
#         s_no_dupl.append(sent[i])
#         t_no_dupl.append(add_t[i])
#     else:
#         t_no_dupl[i-idx][1] = add_t[i][1]
#         idx += 1
        

# # for t, s in zip(t_no_dupl, s_no_dupl):
# #     print(t, s)

# sub.write_subtitles("output_subtitles/aligned_out.srt", s_no_dupl, t_no_dupl, start_offset=start)

