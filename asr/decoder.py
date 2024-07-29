import torch
import torchaudio


torch.set_num_threads(1)


class Decoder:
    """
    A class that performs decoding of audio waveforms into text using a CTC decoder.

    Args:
        waveform (torch.Tensor): The input waveform.
        original_sample_rate (int): The sample rate of the input waveform.
        model (torch.nn.Module): The model used for feature extraction.
        labels (list[str]): The list of labels.
        lexicon_filepath (str): The path to the lexicon file.
        tokens_filepath (str): The path to the tokens file.
        lm_filepath (str): The path to the language model file.
        device (str): The device on which to perform the decoding.
        subtitles (str): The subtitles associated with the waveform.
        resample_rate (int, optional): The sample rate to resample the waveform to. Defaults to 16000.
    """

    def __init__(
        self,
        original_sample_rate,
        model,
        device,
        resample_rate=16000,
    ):
        self.original_sample_rate = original_sample_rate
        self.model = model
        self.device = device
        self.resample_rate = resample_rate


    def decode(self, timestep, audio_slice):
        """
        Decodes the given audio slice within the specified timestep.

        Args:
            timestep (dict): The dictionary representing the start and end times of the timestep.
            audio_slice (torch.Tensor): The audio slice to be decoded.

        Returns:
            tuple: A tuple containing the decoded tokens and their corresponding timesteps.
        """

        print(timestep)
        print(timestep["start"])

        start_time = timestep["start"]
        end_time = timestep["end"]
        duration = end_time - start_time



        # Extract the emission tensor from the audio slice
        emission = self._extract_emission(audio_slice, self.model)
        timestep_ratio = duration / emission.shape[1]

        timesteps = [start_time + i * timestep_ratio for i in range(emission.shape[1])]

        # # Perform beam search decoding using the CTC decoder
        # beam_search_result = self.decoder(emission)

        # best_result = None
        # best_score = -10e6

        # # Find the best result with the highest score
        # for result in beam_search_result:
        #     for r in result:
        #         if r.score > best_score:
        #             best_result = r
        #             best_score = r.score

        # tokens = []
        # timesteps = []

        # # Convert the tokens and timesteps to lowercase and calculate the actual timesteps
        # for token, timestep in zip(best_result.tokens, best_result.timesteps):
        #     token = self.labels[token].lower()
        #     timestep = timestep.item()

        #     if len(tokens) > 0:
        #         if token != "|":
        #             tokens.append(token)
        #             timesteps.append(start_time + timestep_ratio * timestep)

        #         elif tokens[-1] != "|":
        #             tokens.append(token)
        #             timesteps.append(start_time + timestep_ratio * timestep)
        #     else:
        #         tokens.append(token)
        #         timesteps.append(start_time + timestep_ratio * timestep)

        # # Remove the start and end tokens and corresponding timesteps
        # tokens = tokens[1:-1]
        # timesteps = timesteps[1:-1]

        return emission, timesteps

    def _extract_emission(self, waveform, model):
        """
        Extracts the emission tensor from the given waveform using the specified model.

        Args:
            waveform (torch.Tensor): The input waveform.
            model (torch.nn.Module): The model used for feature extraction.

        Returns:
            torch.Tensor: The emission tensor.
        """

        # Resample the waveform to the specified sample rate
        waveform = torchaudio.functional.resample(
            waveform, self.original_sample_rate, self.resample_rate
        )
        # Take the mean across channels and add a batch dimension
        waveform = waveform.mean(axis=0).unsqueeze(0).to(self.device)
        # Extract the emission tensor using the model
        emission, _ = model(waveform)
        # Move the emission tensor to the specified device
        emission = emission.to("cpu")

        return emission
