import torch
import torchaudio

from torchaudio.models.decoder import download_pretrained_files

import copy
import argparse

from asr.audio_slicer import AudioSlicer
import asr.decoder as dec
import asr.subtitles as sub
from asr.locator import Locator
from asr.nw_aligner import NWAligner

import numpy as np
import nltk
from unidecode import unidecode
from num2words import num2words

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline as pipe

import re


def load_subtitles(filename, start, end):
    return sub.Subtitles(filename, start=start, end=end)


def create_locator():
    return Locator()


def download_files():
    return download_pretrained_files("librispeech-4-gram")


def configure_torch():
    torch.random.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return device


def load_bundle(bundle_name):
    if bundle_name == "WAV2VEC2_ASR_BASE_960H":
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    if bundle_name == "WAV2VEC2_ASR_LARGE_960H":
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    if bundle_name == "WAV2VEC2_ASR_LARGE_LV60K_960H":
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    if bundle_name == "HUBERT_ASR_LARGE":
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    if bundle_name == "HUBERT_ASR_XLARGE":
        bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE

    return bundle


def load_model(bundle, device):
    model = bundle.get_model().to(device)
    return model


def load_pipeline_whisper(model_name, device):
    pipeline = pipe(
        "automatic-speech-recognition",
        model=f"openai/{model_name}",
        chunk_length_s=0,
        device=device,
    )

    return pipeline


def load_waveform(filename, start, end):
    waveform, sample_rate = torchaudio.load(filename)
    waveform = waveform[:, int(start * sample_rate) : int(end * sample_rate)]
    return waveform, sample_rate


def load_waveform_whisper(filename, start, end):
    waveform, sample_rate = torchaudio.load(filename)
    waveform = waveform[:, int(start * sample_rate) : int(end * sample_rate)]

    # Resample waveform if needed
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    sample_rate = 16000

    return waveform, sample_rate


def create_decoder(
    waveform, sample_rate, model, bundle, device, subs, files, lm_filepath
):
    return dec.Decoder(
        waveform,
        original_sample_rate=sample_rate,
        model=model,
        labels=bundle.get_labels(),
        lexicon_filepath="lexicon.txt",
        tokens_filepath=files.tokens,
        lm_filepath=lm_filepath,
        device=device,
        subtitles=subs,
        resample_rate=bundle.sample_rate,
    )


def create_slicer():
    return AudioSlicer()


def slice_audio(waveform, sample_rate, slicer, vad):
    timesteps, slices = slicer.slice(
        waveform, sample_rate, max_step=5, buffer_region=0.0, vad=vad
    )
    return timesteps, slices


def run(
    subs,
    locator,
    decoder,
    timesteps,
    slices,
    start,
    output_filename_prefix,
    mode="global",
    _score=-1,
    match_score=10,
    mismatch_score=-1,
    open_gap_score=-1,
    extend_gap_score=-1,
    target_end_gap_score=0.0,
    query_end_gap_score=0.0,
):
    nw_aligner = NWAligner(
        subs,
        mode=mode,
        mismatch_score=mismatch_score,
        match_score=match_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
        target_end_gap_score=target_end_gap_score,
        query_end_gap_score=query_end_gap_score,
    )

    decoded_tokens = []
    decoded_timesteps = []
    decoded_subtitles = []
    decoded_times = []
    predicted_subtitles = []
    predicted_times = []
    predicted_indices = [0]
    predicted_time_list = [0]

    all_tokens = []
    all_timesteps = []

    for timestep, audio_slice in zip(timesteps, slices):
        print(timestep)
        tokens, timesteps, emission = decode_timestep(decoder, timestep, audio_slice)
        tokens = copy.deepcopy(tokens)
        timesteps = copy.deepcopy(timesteps)

        file_path = "emission.txt"

        # Open the file in append mode
        with open(file_path, "a") as file:
            for i in range(emission.shape[0]):
                for j in range(emission.shape[1]):
                    # # Convert the list to a string and write it to the file
                    file.write(str(emission[i][j].tolist()) + "\n")

        file_path = "emission_timesteps.txt"

        # Open the file in append mode
        with open(file_path, "a") as file:
            for i in range(emission.shape[0]):
                for j in range(emission.shape[1]):
                    timestamps = timestep["start"] + j * 0.03
                    file.write(str(timestamps) + "\n")

        if len(tokens) > 0:
            all_tokens += tokens + ["|"]
            all_timesteps += timesteps + [timesteps[-1]]

            decoded_tokens.append(tokens)
            decoded_timesteps.append(timesteps)

            decoded_subtitles.append("".join(tokens))
            decoded_times.append([timesteps[0], timesteps[-1]])

            aligned_subtitles, aligned_times = align_tokens(
                nw_aligner, tokens, timesteps
            )

            good = False

            if len(aligned_subtitles) > 0:
                good = locator.next_index(nw_aligner.predicted_words_sentence_idx[0])

            if good:
                for subtitle, time in zip(aligned_subtitles, aligned_times):
                    predicted_subtitles.append(subtitle)
                    predicted_times.append(time)
                    predicted_indices.append(subs.find(subtitle))
                    predicted_time_list.append(time[0])

    predicted_subtitles, predicted_times = remove_duplicates(
        predicted_subtitles, predicted_times
    )

    sub.write_subtitles(
        "output_subtitles/" + output_filename_prefix + "_decoded.srt",
        decoded_subtitles,
        decoded_times,
        start_offset=start,
    )
    sub.write_subtitles(
        "output_subtitles/" + output_filename_prefix + "_aligned_realtime.srt",
        predicted_subtitles,
        predicted_times,
        start_offset=start,
    )


def run_whisper(
    subs, locator, pipeline, timesteps, slices, start, output_filename_prefix
):
    nw_aligner = NWAligner(subs)

    decoded_tokens = []
    decoded_timesteps = []
    decoded_subtitles = []
    decoded_times = []
    predicted_subtitles = []
    predicted_times = []
    predicted_indices = [0]
    predicted_time_list = [0]

    all_tokens = []
    all_timesteps = []

    for timestep, audio_slice in zip(timesteps, slices):
        print(timestep)
        emission = pipeline(audio_slice, batch_size=1, return_timestamps=False)["text"]
        print(emission)
        emission = [
            {"timestamp": (0, timestep["end"] - timestep["start"]), "text": emission}
        ]
        print(emission)
        # Whisper timestamps sometimes have None output so need to handle that to
        # prevent errors
        for i, (emission_timestep_dict) in enumerate(emission):
            if emission_timestep_dict["timestamp"][1] is None:
                emission[i]["timestamp"] = (
                    emission_timestep_dict["timestamp"][0],
                    timestep["end"] - timestep["start"],
                )
            print(
                f"{emission[i]['timestamp'][0] + timestep['start']} --> {emission[i]['timestamp'][1] + timestep['start']} \t {emission[i]['text']}"
            )
        time_offset = timestep["start"]
        tokens, timesteps = interpolate_timestamps(emission)

        tokens = copy.deepcopy(tokens)
        timesteps = copy.deepcopy(timesteps)

        if len(tokens) > 0:
            timesteps = [t + time_offset for t in timesteps]
            all_tokens += tokens + ["|"]
            all_timesteps += timesteps + [timesteps[-1]]

            decoded_tokens.append(tokens)
            decoded_timesteps.append(timesteps)

            decoded_subtitles.append("".join(tokens))
            decoded_times.append([timesteps[0], timesteps[-1]])

            aligned_subtitles, aligned_times = align_tokens(
                nw_aligner, tokens, timesteps
            )

            good = False

            if len(aligned_subtitles) > 0:
                good = locator.next_index(nw_aligner.predicted_words_sentence_idx[0])

            for k, v in zip(aligned_subtitles, aligned_times):
                print(f"{k}\t{v}")

            if good:
                for subtitle, time in zip(aligned_subtitles, aligned_times):
                    predicted_subtitles.append(subtitle)
                    predicted_times.append(time)
                    predicted_indices.append(subs.find(subtitle))
                    predicted_time_list.append(time[0])

    predicted_subtitles, predicted_times = remove_duplicates(
        predicted_subtitles, predicted_times
    )

    sub.write_subtitles(
        "output_subtitles/" + output_filename_prefix + "_decoded.srt",
        decoded_subtitles,
        decoded_times,
        start_offset=start,
    )
    sub.write_subtitles(
        "output_subtitles/" + output_filename_prefix + "_aligned_realtime.srt",
        predicted_subtitles,
        predicted_times,
        start_offset=start,
    )


def decode_timestep(decoder, timestep, audio_slice):
    tokens, timesteps, emission = decoder.decode(timestep, audio_slice)
    return tokens, timesteps, emission


def align_tokens(nw_aligner, tokens, timesteps):
    aligned_subtitles, aligned_times = nw_aligner.transform(tokens, timesteps)
    return aligned_subtitles, aligned_times


def remove_duplicates(predicted_subtitles, predicted_times):
    predicted_subtitles_no_duplicates = []
    predicted_times_no_duplicates = []

    if len(predicted_subtitles) > 0:
        predicted_subtitles_no_duplicates = [predicted_subtitles[0]]
    if len(predicted_times) > 0:
        predicted_times_no_duplicates = [predicted_times[0]]
    idx = 1
    for i in range(1, len(predicted_subtitles)):
        if predicted_subtitles[i - 1] == predicted_subtitles[i]:
            predicted_times_no_duplicates[i - idx][1] = predicted_times[i][1]
            idx += 1
        else:
            predicted_subtitles_no_duplicates.append(predicted_subtitles[i])
            predicted_times_no_duplicates.append(predicted_times[i])

    return predicted_subtitles_no_duplicates, predicted_times_no_duplicates


def run_decoder(
    filepath,
    filename,
    vad,
    bundle_name,
    mode="global",
    mismatch_score=-1,
    match_score=10,
    open_gap_score=-1,
    extend_gap_score=-1,
    target_end_gap_score=0.0,
    query_end_gap_score=0.0,
    extra_filenam_info=None,
):
    SPEECH_FILE = "data/" + filepath + filename + ".wav"

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end

    trimmed_subs = sub.Subtitles(
        "data/" + filepath + filename + ".srt", start=start, end=end
    )
    start = trimmed_subs.subtitles[0].start
    start = start.hours * 3600.0 + start.minutes * 60.0 + start.seconds
    start = 3 * (start / 3)
    end = trimmed_subs.subtitles[-1].end
    end = end.hours * 3600.0 + end.minutes * 60.0 + end.seconds

    print(start, end)

    subs = load_subtitles("data/" + filepath + filename + ".srt", start=0, end=None)
    locator = create_locator()
    files = download_files()
    device = configure_torch()
    bundle = load_bundle(bundle_name)
    model = load_model(bundle, device)

    print(f"Files {files.tokens}")

    waveform, sample_rate = load_waveform(
        SPEECH_FILE,
        start,
        end,
    )
    decoder = create_decoder(
        waveform,
        sample_rate,
        model,
        bundle,
        device,
        subs,
        files,
        "ngram-lm/arpa/" + filename + ".6gram.arpa",
    )
    slicer = create_slicer()
    timesteps, slices = slice_audio(waveform, sample_rate, slicer, vad=vad)

    if vad:
        output_filename_prefix = f"{bundle_name}_vad_{filename}"
    else:
        output_filename_prefix = f"{bundle_name}_no_vad_{filename}"

    if extra_filenam_info:
        output_filename_prefix = output_filename_prefix + extra_filenam_info
    run(
        subs,
        locator,
        decoder,
        timesteps,
        slices,
        start,
        output_filename_prefix=output_filename_prefix,
        mode=mode,
        mismatch_score=mismatch_score,
        match_score=match_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
        target_end_gap_score=target_end_gap_score,
        query_end_gap_score=query_end_gap_score,
    )


def clean_text(text):
    # Combine all subtitle lines into a single string
    clean_text = text
    # Replace newline characters and dashes with spaces
    clean_text = clean_text.replace("\n", " ").replace("-", " ")

    # Convert numbers to words in the subtitle text
    clean_text = " ".join(
        [num2words(s) if s.isdigit() else s for s in clean_text.split()]
    )

    # Remove diacritical marks from the subtitle text
    clean_text = unidecode(clean_text)

    # Remove non-alphabetic characters and extra spaces from the whole subtitle text, and convert to lowercase
    clean_text = "".join([i if (i.isalpha() or i == " ") else "" for i in clean_text])
    clean_text = clean_text.lower()
    clean_text = clean_text.replace(" ", "|")

    return clean_text


def interpolate_timestamps(structure):
    interpolated_structure = []
    for item in structure:
        text = clean_text(item["text"]) + "|"
        start_time, end_time = item["timestamp"]
        letter_count = len(text)  # Count the number of letters instead of words
        time_per_letter = (
            end_time - start_time
        ) / letter_count  # Calculate time per letter

        start_timestamp = start_time
        for letter in text:
            end_timestamp = start_timestamp + time_per_letter
            interpolated_structure.append(
                {"timestamp": (start_timestamp, end_timestamp), "text": letter}
            )
            start_timestamp = end_timestamp

    tokens = [item["text"] for item in interpolated_structure]
    timesteps = [item["timestamp"][0] for item in interpolated_structure]

    if len(tokens) > 0 and tokens[-1] == "|":
        tokens = tokens[:-1]
        timesteps = timesteps[:-1]

    return tokens, timesteps


def run_decoder_whisper(filepath, filename, vad, model_name):
    SPEECH_FILE = "data/" + filepath + filename + ".wav"

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end

    trimmed_subs = sub.Subtitles(
        "data/" + filepath + filename + ".srt", start=start, end=end
    )
    start = trimmed_subs.subtitles[0].start
    start = start.hours * 3600.0 + start.minutes * 60.0 + start.seconds
    start = 3 * (start / 3)
    end = trimmed_subs.subtitles[-1].end
    end = end.hours * 3600.0 + end.minutes * 60.0 + end.seconds

    print(start, end)

    subs = load_subtitles("data/" + filepath + filename + ".srt", start=0, end=None)
    locator = create_locator()
    device = configure_torch()
    pipeline = load_pipeline_whisper(model_name, device)

    waveform, sample_rate = load_waveform_whisper(
        SPEECH_FILE,
        start,
        end,
    )

    slicer = create_slicer()
    timesteps, slices = slice_audio(waveform, sample_rate, slicer, vad=vad)

    if vad:
        output_filename_prefix = f"{model_name}_vad_{filename}"
    else:
        output_filename_prefix = f"{model_name}_no_vad_{filename}"

    processed_slices = []

    for slice in slices:
        slice = slice.mean(axis=0).unsqueeze(0)
        slice = slice.transpose(0, 1).numpy()
        slice = np.squeeze(slice, axis=1)

        processed_slices.append(slice)

    slices = processed_slices

    run_whisper(
        subs,
        locator,
        pipeline,
        timesteps,
        slices,
        start,
        output_filename_prefix=output_filename_prefix,
    )


if __name__ == "__main__":
    vad_on = [True]
    bundle_names = [
        "WAV2VEC2_ASR_BASE_960H",
    ]

    modes = ["global", "local"]
    matchs = list(range(1, 11))
    end_gap_scores = [0, -1]

    mismatch = -1
    open_gap = -1
    extend_gap = -1

    for bundle_name in bundle_names:
        for vad in vad_on:
            for mode in modes:
                mode_info = f"_{mode}"
                for end_gap_score in end_gap_scores:
                    end_gap_info = f"_end_gap_{end_gap_score}"
                    for match in matchs:
                        match_info = f"_match_score_{match}"
                        extra_filename_info = mode_info + end_gap_info + match_info
                        run_decoder(
                            filepath="Samples/",
                            filename="clean_sample_noise_n25db",
                            vad=vad,
                            bundle_name=bundle_name,
                            extra_filenam_info=extra_filename_info,
                            mode=mode,
                            mismatch_score=mismatch,
                            match_score=match,
                            open_gap_score=open_gap,
                            extend_gap_score=extend_gap,
                            target_end_gap_score=end_gap_score,
                            query_end_gap_score=end_gap_score,
                        )

    vad_on = [True, False]
    bundle_names = [
        "WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H",
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_LARGE",
        "HUBERT_ASR_XLARGE",
    ]

    file_types = ["clean_sample_noise", "clean_sample_speech", "clean_sample_music"]
    noise_levels = range(10, 30)

    for bundle_name in bundle_names:
        for vad in vad_on:
            run_decoder(
                filepath="Samples/",
                filename="clean_sample",
                vad=vad,
                bundle_name=bundle_name,
            )

            for file_type in file_types:
                for noise_level in noise_levels:
                    filename = f"{file_type}_n{noise_level}db"
                    print(filename)
                    run_decoder(
                        filepath="Samples/",
                        filename=filename,
                        vad=vad,
                        bundle_name=bundle_name,
                    )

    vad_on = [True, False]
    model_names = [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        # "whisper-large",
        "whisper-large-v2",
    ]

    file_types = ["clean_sample_noise", "clean_sample_speech", "clean_sample_music"]
    noise_levels = range(10, 30)

    for model_name in model_names:
        for vad in vad_on:
            run_decoder_whisper(
                filepath="Samples/",
                filename="clean_sample",
                vad=vad,
                model_name=model_name,
            )

            for file_type in file_types:
                for noise_level in noise_levels:
                    filename = f"{file_type}_n{noise_level}db"
                    print(filename)
                    run_decoder_whisper(
                        filepath="Samples/",
                        filename=filename,
                        vad=vad,
                        model_name=model_name,
                    )

    bundle_names = [
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_XLARGE",
    ]

    model_names = [
        "whisper-medium",
    ]

    filenames = ["carnival_of_souls", "horror_express", "jungle_book"]

    for filename in filenames:
        for vad in [True, False]:
            for bundle_name in bundle_names:
                run_decoder(
                    filepath="long_audio/",
                    filename=filename,
                    vad=vad,
                    bundle_name=bundle_name,
                )

            for model_name in model_names:
                run_decoder_whisper(
                    filepath="long_audio/",
                    filename=filename,
                    vad=vad,
                    model_name=model_name,
                )

    # Write emissions to file for use in testing alignment method
    file_path = "emission.txt"

    # Open the file in append mode
    with open(file_path, "w") as file:
        # Convert the list to a string and write it to the file
        file.write("\n")

    file_path = "emission_timesteps.txt"

    # Open the file in append mode
    with open(file_path, "w") as file:
        file.write("\n")

    bundle_names = [
        "HUBERT_ASR_XLARGE",
    ]

    filenames = ["jungle_book"]

    for filename in filenames:
        for vad in [False]:
            for bundle_name in bundle_names:
                run_decoder(
                    filepath="long_audio/",
                    filename=filename,
                    extra_filenam_info="_heatmap",
                    vad=vad,
                    bundle_name=bundle_name,
                )
