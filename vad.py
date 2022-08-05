# import webrtcvad
# vad = webrtcvad.Vad()

# vad.set_mode(1)



# from scipy.io import wavfile
# sample_rate, waveform = wavfile.read('./data/spider_man_short16.wav')


# start = 0*sample_rate
# stop = 20*sample_rate

# frame_duration = 10
# step = int(sample_rate * frame_duration / 1000.0)


# vad_on_off = []
# times = []

# for i in range(start, stop, step):

#     frame = waveform[i:i+step].tobytes()

#     vad_on_off.append(int(vad.is_speech(frame, sample_rate)))
#     times.append(i/sample_rate)


# vad_on = False
# time_ranges = []
# for i in range(len(times)):
#     if not vad_on:
#         if vad_on_off[i] == 1:
#             vad_on = True
#             start_time = times[i]
#         else:
#             vad_on = False

#     else:
#         if vad_on_off[i] == 0:
#             vad_on = False
#             end_time = times[i]
#             time_ranges.append([start_time, end_time])
#         else:
#             vad_on = True

# print(time_ranges)


import torch
torch.set_num_threads(1)


def get_speech_timestamps(wav, sample_rate, device=torch.device("cpu")):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    model.to(device)
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # wav = read_audio('./data/spider_man_short16.wav', sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate, visualize_probs=True, return_seconds=True)

    wav_timesteps = []
    for timestep in speech_timestamps:
        start_idx = int(timestep['start']*sample_rate)
        end_idx = int(timestep['end']*sample_rate)

        wav_timesteps.append(wav[start_idx:end_idx])

    return wav_timesteps
    # print(speech_timestamps)

    # wav = read_audio('./data/spider_man_short16.wav', sampling_rate=16000)
    # speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, visualize_probs=True, return_seconds=True)

    # print(speech_timestamps)