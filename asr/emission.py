from itertools import groupby
from collections import defaultdict
import copy
from os import times_result
from .subtitles import Subtitles
from thefuzz import fuzz

#Emission Class
#Consists of Letters/Words and corresponding times
class Emission():
    def __init__(self):
        pass

    def transform(self):
        pass

class IndexAligner(Emission):
    def __init__(self, subtitles):
        self.subtitles = subtitles
        self.current_index = 0
        self.indices = []
        self.subtitle_ranges = self.enumerate_subtitles(self.subtitles)
    
    def transform(self, tokens, timesteps):
        tokens = copy.deepcopy(tokens)
        timesteps = copy.deepcopy(timesteps)

        index = self._string_match(tokens, timesteps)
        # sentences, sentence_times = self._words_and_times(index, tokens, timesteps)
        
        # print("{}".format(self._word_index_to_sentence_index(index)))
        # return sentences, sentence_times
        index = self._word_index_to_sentence_index(index)
        # print("{}\t{}".format(index, self.current_index))
        self._update_index(index)

        text = "".join(tokens).replace("|", " ")

        max_score = -1
        best_match = ""

        matches = {}

        start = max(0, self.current_index-5)
        end = min(len(self.subtitles.subtitles), self.current_index+20)
        subss = Subtitles("data/spider_man_source.srt", start=start, end=end)

        print("LEN: ", len(self.subtitles.sentences), len(subss.sentences))
        for s in subss.sentences:
            
            ratio = fuzz.ratio(text.lower(), s.lower())
            matches[s.lower()] =  ratio

        matches = [k for k, v in sorted(matches.items(), key=lambda item: item[1])][::-1]

        print(text)
        print(matches[:3])
        # best_match = matches[0]

        indexes = [subss.whole_text.find(match) for match in matches]
        print(indexes)

        indexes = [self._word_index_to_sentence_index(idx, subss) + max(0, self.current_index-5) for idx in indexes]
        print(indexes)

        # consecutive_idx = set()
        # for i, idx in enumerate(indexes[:3]):
        #     for j in indexes[:i] + indexes[i+1:3]:
        #         if abs(idx - j) == 1:
        #             consecutive_idx.add(idx)

        # consecutive_idx = list(consecutive_idx)
        # if len(consecutive_idx) == 0:
        #     i = min(indexes[:3], key=lambda x:abs(x-self.current_index))
        #     consecutive_idx = [i]

        consecutive_idx = []

        if len(indexes) > 0:
            consecutive_idx = [indexes[0]]
        
        
        # print("CON: {}".format(consecutive_idx))
        
        # best_idx = self.subtitles.whole_text.find(best_match)
        # best_idx = self._word_index_to_sentence_index(best_idx)
        # i = self._string_match(tokens, timesteps, subtitles=subss)
        # i = self._word_index_to_sentence_index(i)
        # print("{}\t{}\t{}".format(best_idx, i, best_match))
        print("Found idx: {}\t Located idx: {}\t Consecutive idx: {}".format(index, self.current_index, consecutive_idx))

        
        word_count = len(text.split())
        sentences, sentence_times = [], []

        
        if self.current_index > -1:
            for i in consecutive_idx:
                if i != -1:
                    sentence = self.subtitles.get(i)
                    sentences.append(sentence)
                    sentence_times.append([timesteps[0], timesteps[-1]])

        
        # self.indices[-1] = self.current_index
        
        # print("{}\t{}\t{}".format(index, self.current_index, sentence_times))
        return sentences, sentence_times


    def _update_index(self, index):
        self.indices.append(index)
        if len(self.indices) < 3:
            self.current_index += 1
        else:
            if 0 <= self.indices[-2] - self.indices[-3] < 4\
                and 0 < self.indices[-1] - self.indices[-2] < 4:
                self.current_index = index
            else:
                self.current_index += 1
        
        # print("Found idx: {}\t Located idx: {}\t Consecutive idx: ".format(index, self.current_index, consecutive_idx))
    
    def enumerate_subtitles(self, subtitles):
        # self.subtitles = subtitles
        # self.subtitle_ranges = {}

        subtitle_ranges = {}

        count = 0
        for i, sentence in enumerate(subtitles.sentences):
            subtitle_ranges[sentence] = {"original_text":subtitles.text[i], "range":[count, count + len(sentence)]}
            count += len(sentence) + 1    

        return subtitle_ranges    

    def _string_match(self, tokens, timesteps, subtitles=None):
        text = "".join(tokens).replace("|", " ")

        if subtitles is None:
            subtitles = self.subtitles

        # if len(text.split()) < 1:
        #     return [], []

        # word_times = []
        
        # for k, g in groupby(zip(tokens, timesteps), lambda x: x[0] != "|"):
        #     if k:
        #         c, t = zip(*g)
        #         word_times.append([t[0], t[-1]])

        best_idx = -1
        best_ratio = -10e6
        whole_text_hash = defaultdict(lambda:-1)
        for i, word in enumerate(subtitles.whole_text.split()[::-1]):
            if word not in whole_text_hash:
                whole_text_hash[word] = i
        
        whole_text_hash_inv = defaultdict(lambda:-1)

        for k, v in whole_text_hash.items():
            whole_text_hash_inv[v] = k
        
        encoded_whole_text = [whole_text_hash[word] for word in subtitles.whole_text.split()]
        encoded_text = [whole_text_hash[word] for word in text.split()]   
            
        alignment = align_fast(encoded_text, encoded_whole_text)
        best_match = print_alignment(encoded_text, encoded_whole_text, alignment, inv_hash=whole_text_hash_inv)
        best_idx = subtitles.whole_text.find(best_match)

        return best_idx
        
    def _words_and_times(self, index, tokens, timesteps):
        text = "".join(tokens).replace("|", " ")

        text = "".join(tokens).replace("|", " ")

        if len(text.split()) < 1:
            return [], []

        word_times = []
        
        for k, g in groupby(zip(tokens, timesteps), lambda x: x[0] != "|"):
            if k:
                c, t = zip(*g)
                word_times.append([t[0], t[-1]])

        #Count number of words in text
        word_count = len(text.split())

        sentences = []
        sentence_times = []
        first_sentence_found = False

        if index > -1:
            for sentence, original_and_range, in self.subtitle_ranges.items():
                original_sentence = original_and_range["original_text"]
                sentence_range = original_and_range["range"]

                #Find first matching subtitle
                if sentence_range[0] <= index <= sentence_range[1]:
                    first_sentence_found = True

                #Append sentence to list if it is the first one found
                #or if the number of words extends into the next sentences
                if first_sentence_found and word_count > 0:
                    sentence_word_count = len(sentence.split())
                    max_idx = min(sentence_word_count, word_count)

                    sentences.append(original_sentence)

                    start_time = word_times[0][0]
                    end_time = word_times[max_idx-1][0]

                    sentence_times.append([start_time, end_time])

                    word_times = word_times[max_idx:]
                    word_count -= sentence_word_count

        return sentences, sentence_times
    
    def _word_index_to_sentence_index(self, index, subtitles=None):
        if subtitles is None:
            subtitles = self.subtitles
            subtitle_ranges = self.subtitle_ranges
        
        else:
            subtitle_ranges = self.enumerate_subtitles(subtitles)

        if index > -1:
            for sentence, original_and_range, in subtitle_ranges.items():
                original_sentence = original_and_range["original_text"]
                sentence_range = original_and_range["range"]

                #Find first matching subtitle
                if sentence_range[0] <= index <= sentence_range[1]:
                    return subtitles.find(original_sentence)
        
        return -1



from itertools import product
from collections import deque


def needleman_wunsch(x, y):
    """Run the Needleman-Wunsch algorithm on two sequences.

    x, y -- sequences.

    Code based on pseudocode in Section 3 of:

    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf
    """
    N, M = len(x), len(y)
    s = lambda a, b: int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)

def align_fast(x, y):
    """Align two sequences, maximizing the
    alignment score, using the Needleman-Wunsch
    algorithm.

    x, y -- sequences.
    """
    return needleman_wunsch(x, y)

def print_alignment(x, y, alignment, inv_hash):
    seq = ["|"]
    max_len = -1

    for i, j in alignment:
        if i is None:
            if seq[-1] != "|":
                seq.append("|")
            
        else:
            if x[i] != -1:
                seq.append(inv_hash[x[i]] + " ")
    
    seq = "".join(seq).split("|")
    max_len_idx = -1

    for i, s in enumerate(seq):
        if len(s) > max_len:
            max_len = len(s)
            max_len_idx = i

    return seq[max_len_idx]