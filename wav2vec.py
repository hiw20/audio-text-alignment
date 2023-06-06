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


def load_subtitles(filename, start, end):
    return sub.Subtitles(filename, start=start, end=end)


def create_locator():
    return Locator()


def download_files():
    return download_pretrained_files("librispeech-4-gram")


def configure_torch():
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def load_waveform(filename, start, end, bundle, device):
    model = bundle.get_model().to(device)
    waveform, sample_rate = torchaudio.load(filename)
    waveform = waveform[:, int(start * sample_rate) : int(end * sample_rate)]
    return waveform, sample_rate, model


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


def slice_audio(waveform, sample_rate, slicer):
    timesteps, slices = slicer.slice(
        waveform, sample_rate, max_step=3, buffer_region=0.0, vad=False
    )
    return timesteps, slices


def run(subs, locator, decoder, timesteps, slices, start, output_filename_prefix):
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
        tokens, timesteps = decode_timestep(decoder, timestep, audio_slice)
        tokens = copy.deepcopy(tokens)
        timesteps = copy.deepcopy(timesteps)

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

    nw_aligner = NWAligner(subs)
    aligned_subtitles, aligned_times = nw_aligner.transform(all_tokens, all_timesteps)
    sub.write_subtitles(
        "output_subtitles/" + output_filename_prefix + "_aligned_after.srt",
        aligned_subtitles,
        aligned_times,
        start_offset=start,
    )


def decode_timestep(decoder, timestep, audio_slice):
    tokens, timesteps = decoder.decode(timestep, audio_slice)
    return tokens, timesteps


def align_tokens(nw_aligner, tokens, timesteps):
    aligned_subtitles, aligned_times = nw_aligner.transform(tokens, timesteps)
    return aligned_subtitles, aligned_times


def remove_duplicates(predicted_subtitles, predicted_times):
    predicted_subtitles_no_duplicates = [predicted_subtitles[0]]
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


def main(filepath, filename):
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
    bundle = load_bundle("HUBERT_ASR_LARGE")
    waveform, sample_rate, model = load_waveform(
        SPEECH_FILE, start, end, bundle, device
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
    timesteps, slices = slice_audio(waveform, sample_rate, slicer)

    run(
        subs,
        locator,
        decoder,
        timesteps,
        slices,
        start,
        output_filename_prefix=filename,
    )


if __name__ == "__main__":
    main(filepath="Samples/", filename="clean_sample")
    main(filepath="Samples/", filename="clean_sample_noise_n29db")
    main(filepath="Samples/", filename="clean_sample_noise_n28db")
    main(filepath="Samples/", filename="clean_sample_noise_n27db")
    main(filepath="Samples/", filename="clean_sample_noise_n26db")
    main(filepath="Samples/", filename="clean_sample_noise_n25db")
    main(filepath="Samples/", filename="clean_sample_noise_n24db")
    main(filepath="Samples/", filename="clean_sample_noise_n23db")
    main(filepath="Samples/", filename="clean_sample_noise_n22db")
    main(filepath="Samples/", filename="clean_sample_noise_n21db")
    main(filepath="Samples/", filename="clean_sample_noise_n20db")
    main(filepath="Samples/", filename="clean_sample_noise_n19db")
    main(filepath="Samples/", filename="clean_sample_noise_n18db")
    main(filepath="Samples/", filename="clean_sample_noise_n17db")
    main(filepath="Samples/", filename="clean_sample_noise_n16db")
    main(filepath="Samples/", filename="clean_sample_noise_n15db")
    main(filepath="Samples/", filename="clean_sample_noise_n14db")
    main(filepath="Samples/", filename="clean_sample_noise_n13db")
    main(filepath="Samples/", filename="clean_sample_noise_n12db")
    main(filepath="Samples/", filename="clean_sample_noise_n11db")
    main(filepath="Samples/", filename="clean_sample_noise_n10db")

    main(filepath="Samples/", filename="clean_sample_sotu_n29db")
    main(filepath="Samples/", filename="clean_sample_sotu_n28db")
    main(filepath="Samples/", filename="clean_sample_sotu_n27db")
    main(filepath="Samples/", filename="clean_sample_sotu_n26db")
    main(filepath="Samples/", filename="clean_sample_sotu_n25db")
    main(filepath="Samples/", filename="clean_sample_sotu_n24db")
    main(filepath="Samples/", filename="clean_sample_sotu_n23db")
    main(filepath="Samples/", filename="clean_sample_sotu_n22db")
    main(filepath="Samples/", filename="clean_sample_sotu_n21db")
    main(filepath="Samples/", filename="clean_sample_sotu_n20db")
    main(filepath="Samples/", filename="clean_sample_sotu_n19db")
    main(filepath="Samples/", filename="clean_sample_sotu_n18db")
    main(filepath="Samples/", filename="clean_sample_sotu_n17db")
    main(filepath="Samples/", filename="clean_sample_sotu_n16db")
    main(filepath="Samples/", filename="clean_sample_sotu_n15db")
    main(filepath="Samples/", filename="clean_sample_sotu_n14db")
    main(filepath="Samples/", filename="clean_sample_sotu_n13db")
    main(filepath="Samples/", filename="clean_sample_sotu_n12db")
    main(filepath="Samples/", filename="clean_sample_sotu_n11db")
    main(filepath="Samples/", filename="clean_sample_sotu_n10db")

    main(filepath="Samples/", filename="clean_sample_music_n29db")
    main(filepath="Samples/", filename="clean_sample_music_n28db")
    main(filepath="Samples/", filename="clean_sample_music_n27db")
    main(filepath="Samples/", filename="clean_sample_music_n26db")
    main(filepath="Samples/", filename="clean_sample_music_n25db")
    main(filepath="Samples/", filename="clean_sample_music_n24db")
    main(filepath="Samples/", filename="clean_sample_music_n23db")
    main(filepath="Samples/", filename="clean_sample_music_n22db")
    main(filepath="Samples/", filename="clean_sample_music_n21db")
    main(filepath="Samples/", filename="clean_sample_music_n20db")
    main(filepath="Samples/", filename="clean_sample_music_n19db")
    main(filepath="Samples/", filename="clean_sample_music_n18db")
    main(filepath="Samples/", filename="clean_sample_music_n17db")
    main(filepath="Samples/", filename="clean_sample_music_n16db")
    main(filepath="Samples/", filename="clean_sample_music_n15db")
    main(filepath="Samples/", filename="clean_sample_music_n14db")
    main(filepath="Samples/", filename="clean_sample_music_n13db")
    main(filepath="Samples/", filename="clean_sample_music_n12db")
    main(filepath="Samples/", filename="clean_sample_music_n11db")
    main(filepath="Samples/", filename="clean_sample_music_n10db")
