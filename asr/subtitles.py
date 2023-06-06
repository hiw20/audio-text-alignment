import pysrt
import nltk
from unidecode import unidecode
from num2words import num2words
import datetime

nltk.download("punkt")

import pysrt
from num2words import num2words
from unidecode import unidecode
import datetime


class Subtitles:
    def __init__(self, filepath, start=0, end=None):
        # Initialize the Subtitles object with a filepath and optional start and end parameters
        self.subtitles = self._load_subtitles(filepath)
        # Check if start is greater than 0 or end is not None
        if start > 0 or end is not None:
            # Update start and end to be within the range of subtitles
            start = min(start, len(self.subtitles))
            end = min(end, len(self.subtitles))

            # Slice the subtitles list to include only the specified range
            self.subtitles = self.subtitles[start:end]

        # Clean the subtitles and store the cleaned data in separate attributes
        cleaned_subtitles = self._clean_subtitles(self.subtitles)
        self.text = cleaned_subtitles["subtitles_text"]
        self.sentences = cleaned_subtitles["subtitles_sentences"]
        self.whole_text = cleaned_subtitles["subtitles_whole_text"]
        self.unique_words = cleaned_subtitles["subtitles_unique_words"]

        # Extract the start and end times of each subtitle and store them in a separate list
        self.times = [
            [sub.start.to_time(), sub.end.to_time()] for sub in self.subtitles
        ]

        # Create two dictionaries to map subtitle texts to their corresponding indices and vice versa
        self.enumerated_subtitles = {}
        self.inv_enumerated_subtitles = {}
        for i, sub in enumerate(self.subtitles):
            self.enumerated_subtitles[sub.text] = i
            self.inv_enumerated_subtitles[i] = sub.text

    def _load_subtitles(self, filepath):
        # Load subtitles from a given filepath using the pysrt library
        subtitles = pysrt.open(filepath)

        return subtitles

    def _clean_subtitles(self, subtitles):
        # Clean the subtitles by removing unnecessary characters and converting numbers to words
        subtitles_text = []

        # Iterate through each subtitle line
        for line in subtitles:
            # Concatenate the characters in the subtitle text
            subtitles_text.append("".join([i for i in line.text]))

        # Combine all subtitle lines into a single string
        subtitles_whole_text = " ".join(subtitles_text)
        # Replace newline characters and dashes with spaces
        subtitles_whole_text = subtitles_whole_text.replace("\n", " ").replace("-", " ")

        # Convert numbers to words in the subtitle text
        subtitles_whole_text = " ".join(
            [num2words(s) if s.isdigit() else s for s in subtitles_whole_text.split()]
        )

        # Remove diacritical marks from the subtitle text
        subtitles_whole_text = unidecode(subtitles_whole_text)

        subtitles_sentences = []

        # Iterate through each subtitle line
        for line in subtitles_text:
            # Replace newline characters and dashes with spaces
            line = line.replace("\n", " ").replace("-", " ")

            # Convert numbers to words in the subtitle line
            line = " ".join([num2words(s) if s.isdigit() else s for s in line.split()])
            # Remove diacritical marks from the subtitle line
            line = unidecode(line)

            # Remove non-alphabetic characters and extra spaces, and convert to lowercase
            line = [i if (i.isalpha() or i == " ") else "" for i in line]
            line = "".join(line).lower()
            subtitles_sentences.append(line)

        # Remove non-alphabetic characters and extra spaces from the whole subtitle text, and convert to lowercase
        subtitles_whole_text = "".join(
            [i if (i.isalpha() or i == " ") else "" for i in subtitles_whole_text]
        )
        subtitles_whole_text = subtitles_whole_text.lower()

        subtitles_unique_words = set()

        # Iterate through each word in the subtitle text
        for word in subtitles_whole_text.split():
            # Add each unique word to the set
            subtitles_unique_words.add(word)

        return {
            "subtitles_text": subtitles_text,
            "subtitles_sentences": subtitles_sentences,
            "subtitles_whole_text": subtitles_whole_text,
            "subtitles_unique_words": subtitles_unique_words,
        }

    def find(self, subtitle):
        # Find the index of a given subtitle in the enumerated subtitles dictionary
        if subtitle in self.enumerated_subtitles:
            return self.enumerated_subtitles[subtitle]
        else:
            return -1

    def get(self, index):
        # Get the subtitle text corresponding to a given index in the enumerated subtitles dictionary
        if 0 < index < len(self.enumerated_subtitles):
            self.inv_enumerated_subtitles

            return self.inv_enumerated_subtitles[index]

        return ""

    def word_idx_to_sub(self, index):
        # Map a word index to its corresponding subtitle text
        index_map = {}

        idx = 0
        # Iterate over each subtitle and its corresponding sentence
        for subt, sentence in zip(self.subtitles, self.sentences):
            # Get the number of words in the sentence
            num_words = len(sentence.split())
            # Map each word index to the subtitle text
            for i in range(num_words):
                index_map[idx + i] = subt.text

            idx += num_words

        return index_map[index]

    def word_idx_to_sub_idx(self, index):
        # Map a word index to its corresponding subtitle index
        index_map = {}

        idx = 0
        # Iterate over each sentence in the subtitles
        for sentence_idx in range(len(self.sentences)):
            # Get the subtitle and its corresponding sentence
            subt = self.subtitles[sentence_idx]
            sentence = self.sentences[sentence_idx]
            # Get the number of words in the sentence
            num_words = len(sentence.split())
            # Map each word index to the subtitle index
            for i in range(num_words):
                index_map[idx + i] = sentence_idx

            idx += num_words

        return index_map[index]

    def sub_idx_to_word_idx(self, index):
        # Map a subtitle index to its corresponding word index
        index_map = {}

        idx = 0
        # Iterate over each sentence in the subtitles
        for sentence_idx in range(len(self.sentences)):
            # Get the subtitle and its corresponding sentence
            subt = self.subtitles[sentence_idx]
            sentence = self.sentences[sentence_idx]
            # Get the number of words in the sentence
            num_words = len(sentence.split())
            # Map the subtitle index to the starting word index
            index_map[sentence_idx] = idx
            idx += num_words

        return index_map[index]


def write_subtitles(filepath, subtitles, times, start_offset=0.0):
    # Write subtitles to a file given the subtitle texts, times, and an optional start offset
    with open(filepath, "w") as file:
        for i, subtitle_times in enumerate(zip(subtitles, times)):
            subtitle, times = subtitle_times

            start_time = str(
                datetime.timedelta(
                    hours=times[0] / 3600.0 + 0.000000001 + start_offset / 3600.0
                )
            )[:-3]
            end_time = str(
                datetime.timedelta(
                    hours=times[1] / 3600.0 + 0.000000001 + start_offset / 3600.0
                )
            )[:-3]

            file.write(start_time + " --> " + end_time + "\n")

            file.write(subtitle + "\n")
            file.write("\n")


if __name__ == "__main__":
    subs = Subtitles("data/spider_man_source.srt")
    for times in subs.times:
        print(times)
