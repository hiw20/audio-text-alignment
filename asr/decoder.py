import torch
from torchaudio.models.decoder import ctc_decoder
import torchaudio

torch.set_num_threads(1)

class Decoder():
    def __init__(self, waveform,\
                        original_sample_rate,\
                        model,\
                        labels,\
                        lexicon_filepath,\
                        tokens_filepath,\
                        lm_filepath,\
                        device,
                        subtitles,\
                        resample_rate=16000):

        self.waveform = waveform
        self.original_sample_rate = original_sample_rate
        self.model = model
        self.labels = labels
        self.lexicon_filepath = lexicon_filepath
        self.tokens_filepath = tokens_filepath
        self.lm_filepath = lm_filepath
        self.device = device
        self.subtitles = subtitles
        self.resample_rate = resample_rate

        LM_WEIGHT = 2
        WORD_SCORE = 0

        self.decoder = ctc_decoder(
            lexicon=self.lexicon_filepath,
            tokens=self.tokens_filepath,
            lm=self.lm_filepath,
            nbest=1,
            beam_size=32,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
            beam_size_token=None
        )

        # self.decoder = ctc_decoder(
        #     lexicon=self.lexicon_filepath,
        #     tokens=self.tokens_filepath,
        #     nbest=1,
        #     beam_size=1024,
        #     word_score=WORD_SCORE,
        #     beam_size_token=None
        # )

    
    def decode(self, timestep, slice):

        # joined_sentences = ". ".join(self.subtitles.sentences)
        # vad_timesteps, vad_slices = self._timestamps(self.waveform, self.sample_rate, self.device)
        # print(timestep)
        start_time = timestep["start"]
        end_time = timestep["end"]
        duration = end_time - start_time

        emission = self._emission(slice, self.model)
        timestep_ratio = duration / emission.shape[1]

        beam_search_result = self.decoder(emission)

        # for result in beam_search_result:
        #     for r in result:
        #         print("{}\t{}".format(r.score, r.words))

        best_result = None
        best_score = -10e6

        for result in beam_search_result:
            for r in result:
                if r.score > best_score:
                    best_result = r
                    best_score = r.score

        tokens = []
        timesteps = []

        for token, timestep in zip(best_result.tokens, best_result.timesteps):
            token = self.labels[token].lower()
            timestep = timestep.item()
            
            if len(tokens) > 0:
                if token != "|":
                    tokens.append(token)
                    timesteps.append(start_time + timestep_ratio * timestep)

                elif tokens[-1] != "|":
                    tokens.append(token)
                    timesteps.append(start_time + timestep_ratio * timestep)
            else:
                tokens.append(token)
                timesteps.append(start_time + timestep_ratio * timestep)
        
        tokens = tokens[1:-1]
        timesteps = timesteps[1:-1]

        return tokens, timesteps   

        # for timestep, slice in zip(vad_timesteps, vad_slices):
        #     print(timestep)
        #     start_time = timestep["start"]
        #     end_time = timestep["end"]
        #     duration = end_time - start_time

        #     emission = self._emission(slice, self.model)
        #     timestep_ratio = duration / emission.shape[1]

        #     beam_search_result = decoder(emission)

        #     best_result = None
        #     best_score = -10e6

        #     for result in beam_search_result:
        #         for r in result:
        #             if r.score > best_score:
        #                 best_result = r
        #                 best_score = r.score

        #     tokens = []
        #     timesteps = []

        #     for token, timestep in zip(best_result.tokens, best_result.timesteps):
        #         token = self.labels[token].lower()
        #         timestep = timestep.item()
                
        #         if len(tokens) > 0:
        #             if token != "|":
        #                 tokens.append(token)
        #                 timesteps.append(start_time + timestep_ratio * timestep)

        #             elif tokens[-1] != "|":
        #                 tokens.append(token)
        #                 timesteps.append(start_time + timestep_ratio * timestep)
        #         else:
        #             tokens.append(token)
        #             timesteps.append(start_time + timestep_ratio * timestep)
            
        #     tokens = tokens[1:-1]
        #     timesteps = timesteps[1:-1]

        #     yield tokens, timesteps    
    

    def _emission(self, waveform, model):
        waveform = torchaudio.functional.resample(waveform, self.original_sample_rate, self.resample_rate)
        waveform = waveform.mean(axis=0).unsqueeze(0)
        emission, _ = model(waveform)
        emission = emission.to('cpu')

        return emission

    # def _timestamps(self, waveform, sample_rate, device, vad=True, max_step=3):
    #     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    #     model.to(device)
    #     (get_speech_timestamps, _, _, _, _) = utils

    #     if vad:
    #         vad_timesteps = get_speech_timestamps(waveform,\
    #             model,\
    #             sampling_rate=sample_rate,\
    #             visualize_probs=True,\
    #             return_seconds=True
    #         )

    #         timesteps = []

    #         for timestep in vad_timesteps:
    #             start = timestep["start"]
    #             stop = timestep["end"]
    #             step = max_step

    #             for i in np.arange(start, stop, step):
    #                 timesteps.append({'start': i, 'end': i + step})

    #     else:
    #         timesteps = []

    #         start = 0
    #         stop = int(waveform.shape[1]/sample_rate)
    #         step = max_step

    #         for i in range(start, stop, step):
    #             timesteps.append({'start': i, 'end': i + step})

    #     slices = []
    #     ammended_timesteps = []
    #     for timestep in timesteps:
    #         lookback_lookahead = 0.0
    #         start_time = max(0, timestep['start'] - lookback_lookahead)
    #         end_time = min(waveform.shape[1],timestep['end'] + lookback_lookahead)

    #         ammended_timesteps.append({'start': start_time, 'end': end_time})

    #         start_idx = int(start_time*sample_rate)
    #         end_idx =  int(end_time*sample_rate)

    #         slices.append(waveform[:, start_idx:end_idx])

    #     return ammended_timesteps, slices