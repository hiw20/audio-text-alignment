import subtitles as sub
import difflib
from difflib import SequenceMatcher
from itertools import groupby
from collections import defaultdict
import copy


class Aligner():
    def __init__(self, subtitles):
        self.subtitles = subtitles
        self.subtitle_ranges = {}

        count = 0
        for i, sentence in enumerate(self.subtitles.sentences):
            self.subtitle_ranges[sentence] = {"original_text":self.subtitles.text[i], "range":[count, count + len(sentence)]}
            count += len(sentence) + 1

        # print(self.subtitles.whole_text)
        
        # for k,v  in self.subtitle_ranges.items():
        #     print("{}\n{}: {}\n{}: {}".format(self.subtitles.whole_text.find(k),\
        #                                         k, len(k),\
        #                                         v["original_text"], v["range"]))
        #     print()

        self.unique_subtitles = []

    def enumerate_subtitles(self, subtitles):
        self.subtitles = subtitles
        self.subtitle_ranges = {}

        count = 0
        for i, sentence in enumerate(self.subtitles.sentences):
            self.subtitle_ranges[sentence] = {"original_text":self.subtitles.text[i], "range":[count, count + len(sentence)]}
            count += len(sentence) + 1
        



    def align(self, tokens, timesteps, previous_tokens=[], previous_timesteps=[], unique=False, subtitles=None):
        if subtitles is not None:
            self.enumerate_subtitles(subtitles)
        # print(tokens)
        # previous_tokens = copy.deepcopy(previous_tokens)
        # previous_timesteps = copy.deepcopy(previous_timesteps)
        # previous_tokens.extend(tokens)
        # previous_timesteps.extend(timesteps)
        # tokens = previous_tokens
        # timesteps = previous_timesteps
        # print(tokens)

        # if len(timesteps) > 1:
        #     print("PRE STEPS: {}".format([timesteps[0], timesteps[-1]]))

        tokens = copy.deepcopy(tokens)
        timesteps = copy.deepcopy(timesteps)

        sentences, sentence_times = self._string_match(tokens, timesteps, unique)
        # if len(timesteps) > 1:
        #     print("POST STEPS: {}".format(sentence_times))
        # print("".join(tokens).replace("|", " "))
        # print(sentences)

        
        return sentences, sentence_times

    def _string_match(self, tokens, timesteps, unique=True):
        text = "".join(tokens).replace("|", " ")
        text_lenth = len(text)

        if len(text.split()) < 1:
            return [], []

        word_times = []
        
        for k, g in groupby(zip(tokens, timesteps), lambda x: x[0] != "|"):
            if k:
                c, t = zip(*g)
                word_times.append([t[0], t[-1]])

        best_idx = -1
        best_ratio = -10e6
        whole_text_hash = defaultdict(lambda:-1)
        for i, word in enumerate(self.subtitles.whole_text.split()[::-1]):
            if word not in whole_text_hash:
                whole_text_hash[word] = i
        
        whole_text_hash_inv = defaultdict(lambda:-1)

        for k, v in whole_text_hash.items():
            whole_text_hash_inv[v] = k
        
        encoded_whole_text = [whole_text_hash[word] for word in self.subtitles.whole_text.split()]
        encoded_text = [whole_text_hash[word] for word in text.split()]
        # print(encoded_text)
        

            
        alignment = align_fast(encoded_text, encoded_whole_text)
        best_match = print_alignment(encoded_text, encoded_whole_text, alignment, inv_hash=whole_text_hash_inv)
        best_idx = self.subtitles.whole_text.find(best_match)

        # print("In: {}, Out: {}, Match: {}".format(text, best_match, self.subtitles.whole_text[best_idx:best_idx+len(text)]))
        # best_idx = self.subtitles.whole_text.find(best_match)
        # #Find best match for text in subtitles
        # for i in range(0, len(self.subtitles.whole_text), 1):
        #     string_to_check = self.subtitles.whole_text[i: i+text_lenth]
        #     ratio = SequenceMatcher(None, text, string_to_check).quick_ratio()

        #     if ratio > best_ratio:
        #         best_ratio = ratio
        #         best_idx = i
        
        #Count number of words in text
        word_count = len(text.split())

        sentences = []
        sentence_times = []
        first_sentence_found = False

        if best_idx > -1:
            for sentence, original_and_range, in self.subtitle_ranges.items():
                original_sentence = original_and_range["original_text"]
                sentence_range = original_and_range["range"]

                #Find first matching subtitle
                if sentence_range[0] <= best_idx <= sentence_range[1]:
                    print("In: {}\n Out: {}\n Match: {}\n Sentence: {}".format(text, best_match, self.subtitles.whole_text[best_idx:best_idx+len(text)], sentence))
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

                    # print("OG SENTENCE: {}".format(original_sentence))
                    # print(sentences)
                    
        return sentences, sentence_times

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

    # print(max_len_idx, seq[max_len_idx])

    return seq[max_len_idx]
    # print(seq(max_len_idx))
    # print(" ".join(
    #     "_" if i is None else inv_hash[x[i]] for i, _ in alignment
    # ))

    # print("".join(
    #     "" if j is None else inv_hash[y[j]] for _, j in alignment
    # ))
