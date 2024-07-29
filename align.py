import torch
import torchaudio
from asr.decoder import Decoder
from asr.tokeniser import tokenize
from asr.audio_slicer import AudioSlicer
import numpy as np

class Aligner:
    def __init__(self, audio_file, text_file, bundle_name, device):
        self.audio_file = audio_file
        self.text_file = text_file

        bundle = self.load_bundle(bundle_name)
        self.model = self.load_model(bundle, device)

        self.waveform, self.sample_rate = self.load_waveform(audio_file, 0, 10000)

        self.asr_model = Decoder(self.sample_rate,
        self.model,
        device,
        resample_rate=16000)


        self.slicer = AudioSlicer()  # Adjust duration as needed

    def slice_audio(self, waveform, sample_rate, slicer, vad):
        timesteps, slices = slicer.slice(
            waveform, sample_rate, max_step=30, buffer_region=0.0, vad=vad
        )
        return timesteps, slices

    def load_bundle(self, bundle_name):
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


    def load_model(self, bundle, device):
        model = bundle.get_model().to(device)
        return model

    def load_data(self):
        with open(self.text_file, 'r') as f:
            self.text = f.read()
    
    def load_waveform(self, filename, start, end):
        waveform, sample_rate = torchaudio.load(filename, format="mp3")
        waveform = waveform[:, int(start * sample_rate) : int(end * sample_rate)]
        return waveform, sample_rate

    def preprocess(self):
        self.text_tokens = tokenize(self.text)  # Tokenize text using your chosen method

    def decode_audio(self):
        self.timesteps, self.audio_segments = self.slicer.slice(self.waveform, self.sample_rate, max_step=30, buffer_region=0.0, vad=True)

        self.emission_matrix = []  # Store emission probabilities
        self.emission_timesteps = []

        for timestep, segment in zip(self.timesteps, self.audio_segments):
            
            emission, timesteps = self.asr_model.decode(timestep, segment)
            self.emission_matrix.append(emission.detach().numpy())
            self.emission_timesteps.append(timesteps)

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
                        timestamps = timesteps[j]
                        file.write(str(timestamps) + "\n")

        # # Combine emissions and timesteps if needed for alignment
        # self.emission_matrix = np.concatenate(self.emission_matrix, axis=0)
        # self.emission_timesteps = np.concatenate(self.emission_timesteps, axis=0)

    # def save_emissions(self, output_file_prefix="output/emissions"):
    #     # Save emission probabilities to text file
    #     np.savetxt(f"{output_file_prefix}.txt", self.emission_matrix, delimiter=" ")

    #     # Save emission timesteps to text file
    #     np.savetxt(f"{output_file_prefix}_timesteps.txt", self.emission_timesteps, delimiter=" ")


    # def align(self):
    #     # Implement your custom alignment algorithm here
    #     # Use self.audio_tokens and self.text_tokens to produce alignments

    # def save_results(self, output_file="output/alignment.txt"):
    #     # Save the alignment results to the specified output file

def configure_torch():
    torch.random.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: {}".format(device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return device

if __name__ == "__main__":

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

    device = configure_torch()
    aligner = Aligner("./Chapter 01 - The Worst Birthday.mp3", "Chapter 01.txt", "WAV2VEC2_ASR_BASE_960H", device)
    aligner.load_data()
    aligner.preprocess()
    aligner.decode_audio()
