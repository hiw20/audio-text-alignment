from scipy import stats
import numpy as np

#Class that indicates where in the script it believes 
#we are
class Locator():
    def __init__(self, start_index=0, end_index=0, min_window=20) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.min_window = min_window
        self.previous_indices = [start_index]
        self.current_index = 0
        self.last_n_idx = []
        self.last_best_index = [0]
        self.predicted_words = []
        self.predicted_word_times = []
        self.predicted_word_indices = []

    def next_index(self, index, tokens=[], timesteps=[]):
        # if len(tokens) > 0:
        #     new_words = "".join(tokens).replace("|", " ").split()
        #     self.predicted_words.extend(new_words)
        #     split_idx = [i for i, s in enumerate(tokens) if s == "|"]

        #     for t in np.split(timesteps, split_idx):
        #         self.predicted_word_times.append([t[0], t[-1]])
        
        # self.predicted_word_indices.extend([-1]*len(new_words))

        self.last_n_idx.append(index)
        if len(self.last_n_idx) < 3:
            self.current_index += 1
        else:
            if 0 < self.last_n_idx[-2] - self.last_n_idx[-3] < 10\
                and 0 < self.last_n_idx[-1] - self.last_n_idx[-2] < 10:
                self.current_index = index
                self.last_best_index.append(index)
                # self.predicted_word_indices[-len(new_words):] = [index]*len(new_words)
            else:
                self.current_index += 1
        
        print("Current Idx: {}".format(self.current_index))
        
        return self.current_index - 5 < index  < self.current_index + 20
        
        # print("Found idx: {}\t Located idx: {}".format(index, self.current_index))
            
        # if len(self.previous_indices) > 5:
        #     res = stats.linregress(list(range(5)), self.previous_indices[-5:])
            
        #     predicted_value = res.slope * 5 + res.intercept
        #     print(index, predicted_value)

        #     if index - predicted_value > 10:
        #         self.previous_indices.append(self.previous_indices[-1] + 1)
        #     elif index - predicted_value < 10:
        #         self.previous_indices.append(self.previous_indices[-1] - 1)
        #     else:
        #         self.previous_indices.append(index)
        # else:
        #     self.previous_indices.append(index)
            
        # self.current_index = self.previous_indices[-1]
        
        # if len(self.previous_indices) < 2:
        #     self.current_index += 1
        
        # else:
        #     diff_check = self.previous_indices[-1] - self.previous_indices[-2]
        #     if 0 < diff_check and diff_check < 4:
        #         self.current_index = self.previous_indices[-1]
        #     else:
        #         # self.current_index = -1 
        #         self.current_index += 1   

        # if len(self.previous_indices) < 10:
        #     self.previous_indices.append(index)
        # else:
        #     # next_index = np.convolve(self.previous_indices[-9:] + [index], np.ones(10)/10, 'same')
        #     next_index = np.mean(self.previous_indices[-9:] + [index]) + 5
        #     self.previous_indices.append(next_index)
        
        # self.current_index = self.previous_indices[-1]


    def search_range(self):
        # [print(x,y) for x,y in zip(self.predicted_words, self.predicted_word_times)]
        if self.current_index == -1:
            return None

        start = min(self.last_best_index, key=lambda x:self.current_index-20)
        end = self.current_index + 20

        start = max(self.start_index, start)
        start = min(self.end_index, start)

        end = max(self.start_index, end)
        end = min(self.end_index, end)

        best_idx = -2
        words = self.predicted_words
        word_times = self.predicted_word_times

        start = 0
        if len(self.predicted_words) > 10:
            for i, index in enumerate(self.predicted_word_indices[:-10:-1]):
                if index != -1:
                    start = index
                    words = words[-10-i:]
                    word_times = word_times[-10-i:]
                    break


        # for i in range(len(self.predicted_word_indices)-1, 0, -1):
        #     if self.predicted_word_indices[i] != self.current_index\
        #         and self.predicted_word_indices[i] != -1:
        #         best_idx = self.predicted_word_indices[i]
        
        # for i in range(i, 0, -1):
        #     if self.predicted_word_indices[i] != best_idx:
        #         start = self.predicted_word_indices[i]
        #         words = self.predicted_words[-i:]
        #         word_times = self.predicted_word_times[-i:]
        #         break
        
        end = self.current_index + 20

        # [print(x,y) for x,y in zip(words, word_times)]


        return {"start":int(start), "end":int(end)},\
                words,\
                word_times
                

        # return {"start":self.start_index, "end":self.end_index}