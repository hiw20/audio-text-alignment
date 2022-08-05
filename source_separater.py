
import torch
import torchaudio

class SourceSeparator:
    def __init__(self):
        self.separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')

    def separate(self, waveform, sample_rate):
        #Load waveform
        # waveform, sample_rate = torchaudio.load(SPEECH_FILE)

        print(waveform.shape)
        # # waveform = waveform.mean(axis=0).unsqueeze(0)
        # waveform = waveform[:, start*sample_rate:end*sample_rate]
        waveform = waveform.reshape(1, 2, -1)
        print(waveform.shape)

        # #Resample waveform if needed
        # if sample_rate != bundle.sample_rate:
        #     waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


        # print(waveform.shape)
        # waveform = waveform.to(device)

        # loading umxhq four target separator
        

        # print(separator.sample_rate, sample_rate)

        # waveform = torch.rand((1, 1, 100000))
        # print(waveform.shape)
        estimates = self.separator(waveform)
        estimates = estimates[0,0,:,:].reshape(2, -1)

        print("Estimates: ", estimates.shape)
        return estimates