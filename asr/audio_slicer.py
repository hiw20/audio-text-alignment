import torch
import torchaudio
import numpy as np

class AudioSlicer():
    def __init__(self) -> None:
        pass

    def slice(self, waveform, sample_rate, max_step=5, buffer_region=0.0, vad=True):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (get_speech_timestamps, _, _, _, _) = utils

        if vad:
            waveform_1_channel = waveform.mean(axis=0).unsqueeze(0)

            print("SAMPLE RATE: ", sample_rate)

            #Resample waveform if needed
            if sample_rate != 16000:
                waveform_1_channel = torchaudio.functional.resample(waveform_1_channel, sample_rate, 16000)

            vad_timesteps = get_speech_timestamps(waveform_1_channel,\
                model,\
                sampling_rate=16000,\
                visualize_probs=True,\
                return_seconds=True
            )

            timesteps = []

            for timestep in vad_timesteps:
                start = timestep["start"]
                stop = timestep["end"]
                step = max_step

                for i in np.arange(start, stop, step):
                    timesteps.append({'start': i, 'end': i + step})


        else:
            timesteps = []

            start = 0
            stop = int(waveform.shape[1]/sample_rate)
            step = max_step

            for i in range(start, stop, step):
                timesteps.append({'start': i, 'end': i + step})

        slices = []
        ammended_timesteps = []
        for timestep in timesteps:
            start_time = max(0, timestep['start'] - buffer_region)
            end_time = min(waveform.shape[1],timestep['end'] + buffer_region)

            ammended_timesteps.append({'start': start_time, 'end': end_time})

            start_idx = int(start_time*sample_rate)
            end_idx =  int(end_time*sample_rate)

            slices.append(waveform[:, start_idx:end_idx])

            # yield {'start': start_time, 'end': end_time}, waveform[:, start_idx:end_idx]

        return ammended_timesteps, slices