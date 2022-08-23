import pysrt
import nltk
from unidecode import unidecode
from num2words import num2words
import datetime

nltk.download('punkt')

class Subtitles:
    def __init__(self, filepath, start=0, end=None):
        self.subtitles = self._load_subtitles(filepath)
        if start > 0 or end is not None:
            # self.subtitles = self.subtitles.slice(starts_after={'seconds': start}, ends_before={'seconds': end})
            start = min(start, len(self.subtitles))
            end = min(end, len(self.subtitles))

            self.subtitles = self.subtitles[start:end]

        cleaned_subtitles = self._clean_subtitles(self.subtitles)
        self.text = cleaned_subtitles["subtitles_text"]
        self.sentences = cleaned_subtitles["subtitles_sentences"]
        self.whole_text = cleaned_subtitles["subtitles_whole_text"]
        self.unique_words = cleaned_subtitles["subtitles_unique_words"]

        self.times = [[sub.start.to_time(), sub.end.to_time()] for sub in self.subtitles]


        self.enumerated_subtitles = {}
        self.inv_enumerated_subtitles = {}
        for i, sub in enumerate(self.subtitles):
            self.enumerated_subtitles[sub.text] = i
            self.inv_enumerated_subtitles[i] = sub.text

    def _load_subtitles(self, filepath):
        subtitles = pysrt.open(filepath)

        return subtitles

    def _clean_subtitles(self, subtitles):
        subtitles_text = []

        for line in subtitles:
            subtitles_text.append("".join([i for i in line.text]))
        
        subtitles_whole_text = " ".join(subtitles_text)
        subtitles_whole_text = subtitles_whole_text.replace("\n", " ")\
                                                    .replace("-", " ")

        subtitles_whole_text = " ".join([num2words(s) if s.isdigit() else s for s in subtitles_whole_text.split()])

        subtitles_whole_text = unidecode(subtitles_whole_text)

        subtitles_sentences = []
        # for sentence in nltk.sent_tokenize(subtitles_whole_text):
        #     sentence = [i if (i.isalpha() or i == " ") else "" for i in sentence]
        #     sentence = "".join(sentence).lower()
        #     subtitles_sentences.append(sentence)
        for line in subtitles_text:
            line = line.replace("\n", " ")\
                        .replace("-", " ")

            line = " ".join([num2words(s) if s.isdigit() else s for s in line.split()])
            line = unidecode(line)
                        
            line = [i if (i.isalpha() or i == " ") else "" for i in line]
            line = "".join(line).lower()
            subtitles_sentences.append(line)
        
        subtitles_whole_text = "".join([i if (i.isalpha() or i == " ") else "" for i in subtitles_whole_text])
        subtitles_whole_text = subtitles_whole_text.lower()

        subtitles_unique_words = set()

        for word in subtitles_whole_text.split():
            subtitles_unique_words.add(word)
            
        return {"subtitles_text": subtitles_text,
                    "subtitles_sentences": subtitles_sentences,
                    "subtitles_whole_text": subtitles_whole_text,
                    "subtitles_unique_words": subtitles_unique_words}
        
    def find(self, subtitle):
        if subtitle in self.enumerated_subtitles:
            return self.enumerated_subtitles[subtitle]
        else:
            return -1
    
    def get(self, index):
        if 0 < index < len(self.enumerated_subtitles):
            self.inv_enumerated_subtitles

            return self.inv_enumerated_subtitles[index]
        
        return ""

    def word_idx_to_sub(self, index):
        index_map = {}

        idx = 0
        for subt, sentence in zip(self.subtitles, self.sentences):
            num_words = len(sentence.split())
            for i in range(num_words):
                index_map[idx+i] = subt.text
            
            idx += num_words
        
        return index_map[index]

    def word_idx_to_sub_idx(self, index):
        index_map = {}

        idx = 0
        for sentence_idx in range(len(self.sentences)):
        # for subt, sentence in zip(self.subtitles, self.sentences):
            subt = self.subtitles[sentence_idx]
            sentence = self.sentences[sentence_idx]
            num_words = len(sentence.split())
            for i in range(num_words):
                index_map[idx+i] = sentence_idx
            
            idx += num_words
        
        return index_map[index]

    def sub_idx_to_word_idx(self, index):
        index_map = {}

        idx = 0
        for sentence_idx in range(len(self.sentences)):
        # for subt, sentence in zip(self.subtitles, self.sentences):
            subt = self.subtitles[sentence_idx]
            sentence = self.sentences[sentence_idx]
            num_words = len(sentence.split())

            index_map[sentence_idx] = idx
            idx += num_words
            
            
        
        return index_map[index]

def write_subtitles(filepath, subtitles, times, start_offset=0.0):
    with open(filepath, "w") as file:
        for i, subtitle_times in enumerate(zip(subtitles, times)):
            subtitle, times = subtitle_times
            # print(subtitle, times)
            file.write(f"{i+1}\n")

            start_time = str(datetime.timedelta(hours=times[0] / 3600.0 + 0.000000001 + start_offset / 3600.0))[:-3]
            end_time = str(datetime.timedelta(hours=times[1] / 3600.0 + 0.000000001 + start_offset / 3600.0))[:-3]

            file.write(start_time + " --> " + end_time + "\n")

            file.write(subtitle + "\n")
            file.write("\n")

if __name__ == "__main__":
    subs = Subtitles("data/spider_man_source.srt")
    for times in subs.times:
        print(times)