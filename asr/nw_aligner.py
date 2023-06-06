from collections import defaultdict
from asr.emission import Emission
import numpy as np


class NWAligner(Emission):
    """
    This class represents an NWAligner, which is used for aligning predicted words with reference words based on the Needleman-Wunsch algorithm.
    """

    def __init__(self, subtitles, predicted_words=[], predicted_word_times=[]):
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

        # Finding the indices where '|' occurs in the tokens list
        split_idx = [i for i, s in enumerate(tokens) if s == "|"]

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

        # Performing the Needleman-Wunsch algorithm to align the predicted and reference words
        pred_align, ref_align = self.needleman_wunsch(
            predicted_words_hashed, reference_words_hashed, gap=1, mismatch=1, match=1
        )

        # Converting the indices from the alignment to their corresponding words
        pred_align = [i if i != -1 else None for i in pred_align]
        ref_align = [i if i != -1 else None for i in ref_align]

        pred_align = [
            inverse_word_hashmap[i] if i is not None else "-" for i in pred_align
        ]
        ref_align = [
            inverse_word_hashmap[i] if i is not None else "-" for i in ref_align
        ]

        # Storing the aligned predicted and reference words
        self.pred_align = pred_align
        self.ref_align = ref_align

    def prepare_output_sentences(self):
        """
        Prepares the output sentences and their corresponding timestamps.

        Returns:
            A tuple containing the transformed sentences and their corresponding timestamps.
        """
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

        # Print the predicted words sentence indices
        print("Indices: {}".format(self.predicted_words_sentence_idx[:]))

        return sentences_no_duplicates, sentence_times_no_duplicates

    def write_alignment_results(self):
        """
        Writes the alignment results to a file.
        """
        with open("output_subtitles/text.txt", "w") as f:
            f.write("")

        with open("output_subtitles/text.txt", "a") as f:
            for i, j in zip(self.pred_align, self.ref_align):
                f.write("{}\t{}\n".format(i, j))

    def needleman_wunsch(self, sequence1, sequence2, match=1, mismatch=1, gap=1):
        # Calculate the lengths of the input sequences
        len_sequence1 = len(sequence1)
        len_sequence2 = len(sequence2)

        # Optimal score at each possible pair of characters.
        scores = np.zeros((len_sequence1 + 1, len_sequence2 + 1))

        # Initialize the first column of the scores matrix with gap penalties
        scores[:, 0] = np.linspace(0, -len_sequence1 * gap, len_sequence1 + 1)

        # Initialize the first row of the scores matrix with zeros
        # Using zeros rather than gap penalties means that there is
        # no penalty for not aligning the start of sequence2 with
        # the start of sequence1. This means we can align a short snippet
        # of text (sequence2) with a longer piece of text (sequence1).
        scores[0, :] = np.zeros(
            scores[0, :].shape
        )  # np.linspace(0, -len_sequence2 * gap, len_sequence2 + 1)

        # Pointers to trace through an optimal alignment.
        pointers = np.zeros((len_sequence1 + 1, len_sequence2 + 1))

        # Set the pointers in the first column to 3, representing a gap in sequence1
        pointers[:, 0] = 3

        # Set the pointers in the first row to 4, representing a gap in sequence2
        pointers[0, :] = 4

        # Temporary scores.
        temp_scores = np.zeros(3)

        # Calculate scores and pointers for each position in the scores matrix
        for i in range(len_sequence1):
            for j in range(len_sequence2):
                # Calculate the score for a match or mismatch
                if sequence1[i] == sequence2[j]:
                    temp_scores[0] = scores[i, j] + match
                else:
                    temp_scores[0] = scores[i, j] - mismatch

                # Calculate the score for a gap in sequence1
                temp_scores[1] = scores[i, j + 1] - gap

                # Calculate the score for a gap in sequence2
                temp_scores[2] = scores[i + 1, j] - gap

                # Determine the maximum score among the three options
                max_score = np.max(temp_scores)

                # Update the scores matrix with the maximum score
                scores[i + 1, j + 1] = max_score

                # Update the pointers matrix based on the maximum score
                if temp_scores[0] == max_score:
                    pointers[i + 1, j + 1] += 2
                if temp_scores[1] == max_score:
                    pointers[i + 1, j + 1] += 3
                if temp_scores[2] == max_score:
                    pointers[i + 1, j + 1] += 4

        # Trace through an optimal alignment.
        # Find the index of the maximum score in the last row of the scores matrix
        max_index = np.where(scores[-1, :] == np.max(scores[-1, :]))[0][0]

        # Initialize the starting positions for backtracking
        i = len_sequence1
        # Setting the start index of j to max_index rather than len_sequence2
        # means that the ebd of sequence2 does not have to align with the end of
        # sequence1.
        j = max_index  # len_sequence2

        # Lists to store the aligned sequences
        aligned_sequence1 = []
        aligned_sequence2 = []

        # Backtrack from the maximum score position to the start position
        while i > 0 or j > 0:
            if pointers[i, j] in [2, 5, 6, 9]:
                # Match or mismatch
                aligned_sequence1.append(sequence1[i - 1])
                aligned_sequence2.append(sequence2[j - 1])
                i -= 1
                j -= 1
            elif pointers[i, j] in [3, 5, 7, 9]:
                # Gap in sequence1
                aligned_sequence1.append(sequence1[i - 1])
                aligned_sequence2.append(None)
                i -= 1
            elif pointers[i, j] in [4, 6, 7, 9]:
                # Gap in sequence2
                aligned_sequence1.append(None)
                aligned_sequence2.append(sequence2[j - 1])
                j -= 1

        # Reverse the aligned sequences to get the correct order
        aligned_sequence1 = aligned_sequence1[::-1]
        aligned_sequence2 = aligned_sequence2[::-1]

        # Return the aligned sequences
        return aligned_sequence1, aligned_sequence2
