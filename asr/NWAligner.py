# from itertools import groupby
from collections import defaultdict
# import copy
# from os import times_result
# from .subtitles import Subtitles
# from thefuzz import fuzz
from asr.emission import Emission
import asr.needleman_wunsch as nw
import numpy as np

class NWAligner(Emission):
    def __init__(self, subtitles, predicted_words=[], predicted_word_times=[]):
        self.complete_subtitles = subtitles
        self.subtitles = subtitles

        self.predicted_words = predicted_words
        self.predicted_word_times = predicted_word_times
        self.predicted_words_sentence_idx = []
        self.reference_words = []

        for word in subtitles.whole_text.split():
            self.reference_words.append(word)
        
        self.number_of_previous_words = 0
        
        # self.max_word_search_length = 100000000

        # self.nw = nw.NW()
    
    # def update_subtitles(self, subtitles, index):
    #     self.subtitles = subtitles

    #     self.number_of_previous_words = self.complete_subtitles.sub_idx_to_word_idx(index)
    #     self.reference_words = []

    #     for word in subtitles.whole_text.split():
    #         self.reference_words.append(word)
        
        
    #     if len(self.predicted_words) > 0:
    #         closest_search_value = min(self.predicted_words_sentence_idx, key=lambda x:abs(x-index))
    #         search_idx = self.predicted_words_sentence_idx.index(closest_search_value)

    #         self.predicted_words = self.predicted_words[search_idx:]
    #         self.predicted_word_times = self.predicted_word_times[search_idx:]
    #         self.predicted_words_sentence_idx = self.predicted_words_sentence_idx[search_idx:]

    def transform(self, tokens, timesteps):
        self.predicted_words = []
        self.predicted_word_times = []
        self.predicted_words_sentence_idx = []
        self.reference_words = []

        for word in self.subtitles.whole_text.split():
            self.reference_words.append(word)
        
        self.number_of_previous_words = 0

        if len(tokens) > 0:
            new_words = "".join(tokens).replace("|", " ").split()
            self.predicted_words.extend(new_words)
            split_idx = [i for i, s in enumerate(tokens) if s == "|"]

            for t in np.split(timesteps, split_idx):
                self.predicted_word_times.append([t[0], t[-1]])
            
            # if len(self.predicted_words) > self.max_word_search_length:
            #     self.predicted_words = self.predicted_words[-self.max_word_search_length:]
            #     self.predicted_word_times = self.predicted_word_times[-self.max_word_search_length:]

            word_hashmap = defaultdict(lambda:-1)
            for i, word in enumerate(self.reference_words[::-1] + self.predicted_words[::-1]):
                if word not in word_hashmap:
                    word_hashmap[word] = i


            inverse_word_hashmap = defaultdict(lambda:-1)

            for k, v in word_hashmap.items():
                inverse_word_hashmap[v] = k

            predicted_words_hashed = [word_hashmap[word] for word in self.predicted_words]
            reference_words_hashed = [word_hashmap[word] for word in self.reference_words]

            pred_align, ref_align = nw.nw(predicted_words_hashed, reference_words_hashed, gap=1, mismatch=1, match=1)
            # pred_align, ref_align = self.nw.nw(predicted_words_hashed, reference_words_hashed, gap=1, mismatch=1, match=1)

            pred_align = [i if i != -1 else None for i in pred_align]
            ref_align = [i if i != -1 else None for i in ref_align]

            pred_align = [inverse_word_hashmap[i] if i is not None else "-" for i in pred_align]
            ref_align = [inverse_word_hashmap[i] if i is not None else "-" for i in ref_align]


            sentence_times = []
            sentences = []

            idx = 0
            idy = self.number_of_previous_words
            for i, word in enumerate(pred_align):
                if i-idx >= len(self.predicted_word_times):
                    break

                if idy >= len(self.reference_words):
                    sentence_times.append(self.predicted_word_times[i-idx])
                    sentences.append("")
                    self.predicted_words_sentence_idx.append(-1)
                    continue

                if pred_align[i] != "-":
                    sentence_times.append(self.predicted_word_times[i-idx])
                    sentences.append(self.complete_subtitles.word_idx_to_sub(idy))
                    self.predicted_words_sentence_idx.append(self.complete_subtitles.word_idx_to_sub_idx(idy))

                if pred_align[i] == "-":
                    idx += 1
                if ref_align[i] != "-":
                    idy += 1

            sentence_times_no_duplicates = [sentence_times[len(sentences) - len(new_words)]]
            sentences_no_duplicates = [sentences[len(sentences) - len(new_words)]]

            idx = len(sentences) - len(new_words) + 1

            for i in range(len(sentences) - len(new_words) + 1, len(sentences)):
                if sentences[i] != sentences_no_duplicates[i-idx]:
                    sentences_no_duplicates.append(sentences[i])
                    sentence_times_no_duplicates.append(sentence_times[i])
                else:
                    sentence_times_no_duplicates[i-idx][1] = sentence_times[i][1]
                    idx += 1
            
            print("Indices: {}".format(self.predicted_words_sentence_idx[:]))
            
            with open("output_subtitles/text.txt", 'w') as f:
                f.write("")

            with open("output_subtitles/text.txt", 'a') as f:
                for i,j in zip(pred_align, ref_align):
                    f.write("{}\t{}\n".format(i, j))
            
            return sentences_no_duplicates, sentence_times_no_duplicates
        
        else:
            return [], []

