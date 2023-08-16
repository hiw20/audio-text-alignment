import torch
import torchaudio
import numpy as np


class AudioSlicer:
    def __init__(self) -> None:
        pass

    def slice(self, waveform, sample_rate, max_step=5, buffer_region=0.0, vad=True):
        """
        Slices an audio waveform into smaller segments based on voice activity detection (VAD) or fixed timesteps.

        Args:
            waveform (torch.Tensor): The audio waveform to be sliced.
            sample_rate (int): The sample rate of the audio waveform.
            max_step (float, optional): The maximum duration (in seconds) for each sliced segment. Defaults to 5.
            buffer_region (float, optional): The duration (in seconds) to include before and after each sliced segment.
                                             Defaults to 0.0.
            vad (bool, optional): Determines whether to use voice activity detection (VAD) for slicing.
                                  If True, VAD will be used; otherwise, fixed timesteps will be used. Defaults to True.

        Returns:
            Tuple: A tuple containing:
                - ammended_timesteps (list): A list of dictionaries containing the start and end times (in seconds)
                                             of each sliced segment, after considering the buffer region.
                - slices (list): A list of sliced segments, where each segment is a torch.Tensor.
        """
        # Load VAD model
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        (get_speech_timestamps, _, _, _, _) = utils

        if vad:
            # Convert multi-channel waveform to mono
            waveform_mono = waveform.mean(axis=0).unsqueeze(0)

            print("SAMPLE RATE: ", sample_rate)

            # Resample waveform if needed
            if sample_rate != 16000:
                waveform_mono = torchaudio.functional.resample(
                    waveform_mono, sample_rate, 16000
                )

            # Perform Voice Activity Detection (VAD)
            vad_timesteps = get_speech_timestamps(
                waveform_mono,
                model,
                sampling_rate=16000,
                visualize_probs=True,
                return_seconds=True,
            )

            timesteps = []

            # Generate timesteps for slicing based on VAD results
            for timestep in vad_timesteps:
                start = timestep["start"]
                stop = timestep["end"]
                step = min(max_step, abs(stop - start))

                for i in np.arange(start, stop, step):
                    begin = i
                    end = min(stop, i + step)
                    timesteps.append({"start": begin, "end": end})

        else:
            timesteps = []

            start = 0
            stop = int(waveform.shape[1] / sample_rate)
            step = max_step

            print(f"Start: {start}, Stop: {stop}, Step: {step}")

            # Generate fixed timesteps for slicing
            for i in range(start, stop, step):
                end = min(stop, i + step)
                timesteps.append({"start": i, "end": end})
                print(f"Start: {i}, Stop: {end}, Step: {i + step}")

        slices = []
        ammended_timesteps = []

        # Slice the waveform based on the timesteps and buffer region
        for timestep in timesteps:
            start_time = max(0, timestep["start"] - buffer_region)
            end_time = min(waveform.shape[1], timestep["end"] + buffer_region)

            ammended_timesteps.append({"start": start_time, "end": end_time})

            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)

            slices.append(waveform[:, start_idx:end_idx])

        return ammended_timesteps, slices
