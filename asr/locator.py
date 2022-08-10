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

    def next_index(self, index):
        self.last_n_idx.append(index)
        if len(self.last_n_idx) < 3:
            self.current_index += 1
        else:
            if self.last_n_idx[-3] == self.last_n_idx[-2] - 1\
                and self.last_n_idx[-2] == self.last_n_idx[-1] - 1:
                self.current_index = index
            else:
                self.current_index += 1
        
        print(index, self.current_index)
            
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
        if self.current_index == -1:
            return None

        start = self.current_index - 10
        end = self.current_index + 50

        start = max(self.start_index, start)
        start = min(self.end_index, start)

        end = max(self.start_index, end)
        end = min(self.end_index, end)

        return {"start":int(start), "end":int(end)}

        # return {"start":self.start_index, "end":self.end_index}