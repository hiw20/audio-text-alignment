from collections import defaultdict
from asr.emission import Emission
import numpy as np
from Bio import pairwise2
from Bio.Align import PairwiseAligner
import os


class NWAligner(Emission):
    """
    This class represents an NWAligner, which is used for aligning predicted words with reference words based on the Needleman-Wunsch algorithm.
    """

    def __init__(
        self,
        subtitles,
        predicted_words=[],
        predicted_word_times=[],
        mode="global",
        match_score=10,
        mismatch_score=-1,
        open_gap_score=-1,
        extend_gap_score=-1,
        target_end_gap_score=0.0,
        query_end_gap_score=0.0,
    ):
        """
        Initializes the NWAligner object.

        Args:
            subtitles: Subtitles object representing the complete set of subtitles.
            predicted_words: List of predicted words.
            predicted_word_times: List of predicted word times.
        """
        self.complete_subtitles = subtitles
        self.subtitles = subtitles

        self.predicted_words = predicted_words
        self.predicted_word_times = predicted_word_times
        self.predicted_words_sentence_idx = []
        self.reference_words = []
        self.number_of_previous_words = 0

        self.history = []
        self.history_length = 15

        self.pred_align = None
        self.ref_align = None

        self.mode = mode
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.open_gap_score = open_gap_score
        self.extend_gap_score = extend_gap_score
        self.target_end_gap_score = target_end_gap_score
        self.query_end_gap_score = query_end_gap_score

    def split_text_into_words(self):
        """
        Splits the whole_text attribute of the subtitles object into a list of words and assigns it to the reference_words attribute.
        """
        self.reference_words = self.subtitles.whole_text.split()

    def transform(self, tokens, timesteps):
        """
        Transforms the predicted words and their corresponding timestamps using the Needleman-Wunsch algorithm.

        Args:
            tokens: List of predicted words.
            timesteps: List of predicted word timestamps.

        Returns:
            A tuple containing the transformed sentences and their corresponding timestamps.
        """
        self.predicted_words = []
        self.predicted_word_times = []
        self.predicted_words_sentence_idx = []
        self.reference_words = []
        self.number_of_previous_words = 0

        self.split_text_into_words()

        if len(tokens) > 0:
            self.process_predicted_words(tokens, timesteps)
            self.align_predicted_and_reference_words()
            (
                sentences_no_duplicates,
                sentence_times_no_duplicates,
            ) = self.prepare_output_sentences()

            self.write_alignment_results()

            return sentences_no_duplicates, sentence_times_no_duplicates
        else:
            return [], []

    def process_predicted_words(self, tokens, timesteps):
        """
        Processes the predicted words and their corresponding timestamps.

        Args:
            tokens: List of predicted words.
            timesteps: List of predicted word timestamps.
        """
        # Joining the predicted words into a single string, replacing '|' with a space, and splitting the string into a list of words
        new_words = "".join(tokens).replace("|", " ").split()
        self.predicted_words.extend(new_words)

        if len(tokens) > 0:
            if tokens[0] == "|":
                tokens = tokens[1:]
                timesteps = timesteps[1:]

        # Finding the indices where '|' occurs in the tokens list
        split_idx = [i for i, s in enumerate(tokens) if s == "|"]

        for idx in split_idx:
            if idx != len(timesteps) - 1:
                timesteps[idx] = timesteps[idx + 1]

        # Splitting the timesteps list based on the indices found above and storing the start and end times for each group of predicted words
        for t in np.split(timesteps, split_idx):
            self.predicted_word_times.append([t[0], t[-1]])

    def create_word_hashmap(self):
        """
        Creates a hashmap to map each unique word in the reference and predicted words list to an index.
        """
        word_hashmap = defaultdict(lambda: -1)
        for i, word in enumerate(
            self.reference_words[::-1] + self.predicted_words[::-1]
        ):
            if word not in word_hashmap:
                word_hashmap[word] = i
        return word_hashmap

    def create_inverse_word_hashmap(self, word_hashmap):
        """
        Creates an inverse hashmap to map the index back to the word.

        Args:
            word_hashmap: Hashmap mapping words to indices.

        Returns:
            Inverse hashmap mapping indices to words.
        """
        inverse_word_hashmap = defaultdict(lambda: -1)
        for k, v in word_hashmap.items():
            inverse_word_hashmap[v] = k
        return inverse_word_hashmap

    def align_predicted_and_reference_words(self):
        """
        Performs the Needleman-Wunsch algorithm to align the predicted and reference words.
        """
        # Creating a hashmap to map each unique word in the reference and predicted words list to an index
        word_hashmap = self.create_word_hashmap()

        # Creating an inverse hashmap to map the index back to the word
        inverse_word_hashmap = self.create_inverse_word_hashmap(word_hashmap)

        # Creating lists of hashed predicted and reference words
        predicted_words_hashed = [word_hashmap[word] for word in self.predicted_words]
        reference_words_hashed = [word_hashmap[word] for word in self.reference_words]
        history_words_hashed = [word_hashmap[word] for word in self.history]

        best_alignment = self.needleman_wunsch(
            reference_seq=reference_words_hashed,
            predicted_seq=history_words_hashed + predicted_words_hashed,
            mode=self.mode,
            mismatch_score=self.mismatch_score,
            match_score=self.match_score,
            open_gap_score=self.open_gap_score,
            extend_gap_score=self.extend_gap_score,
            target_end_gap_score=self.target_end_gap_score,
            query_end_gap_score=self.query_end_gap_score,
        )

        idx_of_pred_words = len(self.history)

        start_idx_of_best_alignment_ref = -1
        start_idx_of_best_alignment_pred = -1
        for ref_idx, pred_idx in zip(
            best_alignment.indices[0], best_alignment.indices[1]
        ):
            if pred_idx >= idx_of_pred_words:
                start_idx_of_best_alignment_ref = ref_idx
                start_idx_of_best_alignment_pred = pred_idx
                break

        ref_align = []
        pred_align = []

        if start_idx_of_best_alignment_ref != -1:
            ref_align = [word for word in self.reference_words]
            pred_align = ["-"] * start_idx_of_best_alignment_ref + self.predicted_words

        self.pred_align = pred_align
        self.ref_align = ref_align

        self.history.extend(self.predicted_words)
        if len(self.history) > self.history_length:
            self.history = self.history[-self.history_length :]

    def prepare_output_sentences(self):
        """
        Prepares the output sentences and their corresponding timestamps.

        Returns:
            A tuple containing the transformed sentences and their corresponding timestamps.
        """
        if len(self.pred_align) == 0:
            return [], []
        sentence_times = []
        sentences = []
        idx = 0
        idy = self.number_of_previous_words

        # Iterate over the indices and words in pred_align
        for i, word in enumerate(self.pred_align):
            # Check if the index is beyond the available predicted word times
            if i - idx >= len(self.predicted_word_times):
                break

            # Check if the index is beyond the available reference words
            if idy >= len(self.reference_words):
                # Append the predicted word time and an empty string to the respective lists
                sentence_times.append(self.predicted_word_times[i - idx])
                sentences.append("")
                self.predicted_words_sentence_idx.append(-1)
                continue

            # Check if the predicted word is not a gap ("-")
            if self.pred_align[i] != "-":
                # Append the predicted word time and the corresponding subtitle to the respective lists
                sentence_times.append(self.predicted_word_times[i - idx])
                sentences.append(self.complete_subtitles.word_idx_to_sub(idy))
                self.predicted_words_sentence_idx.append(
                    self.complete_subtitles.word_idx_to_sub_idx(idy)
                )

            # Increment idx if the predicted word is a gap ("-")
            if self.pred_align[i] == "-":
                idx += 1
            # Increment idy if the reference word is not a gap ("-")
            if self.ref_align[i] != "-":
                idy += 1

        # Initialize lists with the first elements from sentences and sentence_times
        sentence_times_no_duplicates = []
        sentences_no_duplicates = []
        if len(sentence_times) > 0:
            # Initialize lists with the first elements from sentences and sentence_times
            sentence_times_no_duplicates = [
                sentence_times[len(sentences) - len(self.predicted_words)]
            ]
            sentences_no_duplicates = [
                sentences[len(sentences) - len(self.predicted_words)]
            ]

        idx = len(sentences) - len(self.predicted_words) + 1

        # Iterate over the range of indices for sentences excluding the first elements
        for i in range(len(sentences) - len(self.predicted_words) + 1, len(sentences)):
            # Check if the current sentence is different from the previous one
            if sentences[i] != sentences_no_duplicates[i - idx]:
                # Append the current sentence and its corresponding time to the respective lists
                sentences_no_duplicates.append(sentences[i])
                sentence_times_no_duplicates.append(sentence_times[i])
            else:
                # Update the end time of the previous sentence if it's the same as the current one
                sentence_times_no_duplicates[i - idx][1] = sentence_times[i][1]
                idx += 1

        return sentences_no_duplicates, sentence_times_no_duplicates

    def write_alignment_results(self):
        """
        Writes the alignment results to a file.
        """
        folder_path = "output_subtitles"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("output_subtitles/text.txt", "w") as f:
            f.write("")

        with open("output_subtitles/text.txt", "a") as f:
            for i, j in zip(self.pred_align, self.ref_align):
                f.write("{}\t{}\n".format(i, j))

    def needleman_wunsch(
        self,
        reference_seq,
        predicted_seq,
        mode="global",
        match_score=1,
        mismatch_score=-1,
        open_gap_score=-1,
        extend_gap_score=-1,
        target_end_gap_score=0.0,
        query_end_gap_score=0.0,
    ):
        def ints_to_string(int_list):
            int_list = [i if i != 45 else 1114110 for i in int_list]
            unicode_list = [chr(i + 1) for i in int_list]
            str_list = "".join([chr(ord(c)) for c in unicode_list])
            return str_list

        def string_to_ints(str_list):
            unicode_list = [chr(ord(c)) for c in str_list]

            int_list = []
            for i in range(len(unicode_list)):
                ord_c = ord(unicode_list[i])
                if ord_c == 45:
                    int_list.append(-1)
                elif ord_c == 1114111:
                    int_list.append(45)
                else:
                    int_list.append(ord_c - 1)

            return int_list

        reference_seq_converted = ints_to_string(reference_seq)
        predicted_seq_converted = ints_to_string(predicted_seq)

        # Create an aligner object
        aligner = PairwiseAligner()

        # Set the aligner attributes
        aligner.mode = mode
        aligner.match_score = match_score
        aligner.mismatch_score = mismatch_score
        aligner.open_gap_score = open_gap_score
        aligner.extend_gap_score = extend_gap_score
        aligner.target_end_gap_score = target_end_gap_score
        aligner.query_end_gap_score = query_end_gap_score

        # Perform the alignment
        scores = aligner.score(reference_seq_converted, predicted_seq_converted)
        alignments = aligner.align(reference_seq_converted, predicted_seq_converted)

        best_alignment = alignments[0]
        alignment_reference_seq = str(best_alignment[0])
        alignment_predicted_seq = str(best_alignment[1])
        alignment_score = best_alignment.score

        start = next(
            (i for i, element in enumerate(alignment_predicted_seq) if element != "-"),
            -1,
        )
        end = len(alignment_predicted_seq) - next(
            (
                i
                for i, element in enumerate(reversed(alignment_predicted_seq))
                if element != "-"
            ),
            -1,
        )

        return best_alignment
