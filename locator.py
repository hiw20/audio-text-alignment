import numpy as np

class Locator():
    def __init__(self, start_index=0, end_index=0, min_window=5) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.min_window = min_window
        self.previous_indices = [start_index]

    def next_index(self, index):
        if len(self.previous_indices) < self.min_window:
            self.previous_indices.append(index)
        
        elif index > np.mean(self.previous_indices[-self.min_window:]):
            self.previous_indices.append(index)
            # self.previous_indices.append(self.previous_indices[-self.min_window:] + 1)
        # else:
            
        # else:
        #     self.previous_indices = self.previous_indices[:9] + [index]


    def search_range(self):
        if len(self.previous_indices) < self.min_window:
            return {"start":self.start_index, "end":self.end_index}
        else:
            last_indices = self.previous_indices[-self.min_window:]
            std = 0#np.std(self.previous_indices)
            mean = np.mean(last_indices)
            start = mean - std - self.min_window
            end = mean + std + self.min_window

            start = max(self.start_index, start)
            start = min(self.end_index, start)

            end = max(self.start_index, end)
            end = min(self.end_index, end)


            return {"start":int(start), "end":int(end)}
