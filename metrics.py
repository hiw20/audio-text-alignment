from sklearn.metrics import accuracy_score
import subtitles as subs
import matplotlib.pyplot as plt
import numpy as np
import scipy

class Metric:
    def __init__(self):
        pass

    def calculate(self, reference_subtitles, predicted_subtitles):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self):
        pass

    def _time_to_seconds(self, time):
        return time.hour * 3600 + time.minute * 60 + time.second

    def calculate(self, reference_subtitles, predicted_subtitles):
        correct = 0

        ref_hash = {}
        ref_enumerated = {}
        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            ref_hash[i] = [ref_sub.start.to_time(), ref_sub.end.to_time()]
            ref_enumerated[ref_sub.text] = i

        accuracy = []
        ref_x = []
        ref_y = []
        pred_x = []
        pred_y = []

        

        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            ref_x.append(self._time_to_seconds(ref_hash[ref_enumerated[ref_sub.text]][0]))
            ref_y.append(ref_enumerated[ref_sub.text])

        for pred_sub in predicted_subtitles.subtitles:
            if pred_sub.text in ref_enumerated:
                pred_x.append(self._time_to_seconds(pred_sub.start.to_time()))
                pred_y.append(ref_enumerated[pred_sub.text])

                time_period = ref_hash[ref_enumerated[pred_sub.text]]

                if self._time_to_seconds(time_period[0]) < self._time_to_seconds(pred_sub.start.to_time())+3\
                    and self._time_to_seconds(pred_sub.end.to_time()) < self._time_to_seconds(time_period[1])+3:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
            else:
                accuracy.append(0)

        correct = sum(accuracy)
        incorrect = len(accuracy) - correct

        pred_hash = {}
        pred_enumerated = {}
        for i, pred_sub in enumerate(predicted_subtitles.subtitles):
            pred_hash[i] = [pred_sub.start.to_time(), pred_sub.end.to_time()]
            pred_enumerated[pred_sub.text] = i

        for ref_sub in reference_subtitles.subtitles:
            if ref_sub.text not in pred_enumerated:
                accuracy.append(0)
        
        total = len(accuracy)

        # print(pred_enumerated)
        # print(ref_enumerated)
            
        
        # ref_x_zero = ref_x[0]
        # for i in range(len(ref_x)):
        #     ref_x[i] = ref_x[i]# - ref_x_zero
        
        # pred_x_zero = pred_x[0]
        # for i in range(len(pred_x)):
        #     pred_x[i] = pred_x[i]# - pred_x_zero
        
        
        # pred_x, pred_y = self._reject_outliers(pred_x, pred_y)
        
        print("Correct: {}, Incorrect: {}, Missed: {}, Total: {}, Accuracy: {}".format(correct, incorrect, len(accuracy) - correct - incorrect, total, correct/total))
        # print("Ref length: {}, Pred length: {}".format(len(ref_hash), len(pred_hash)))
        plt.plot(ref_x, ref_y, 'r.')
        plt.plot(pred_x, pred_y, 'b.')
        #Save figure
        plt.savefig('plot.png')
    
    def _reject_outliers(self, x, y, m=0.1):
        x = np.array(x)
        y = np.array(y)
        d = np.abs(x - np.median(x))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return x[s<m], y[s<m]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end
    # reference_subtitles = subs.Subtitles("./data/spider_man_source.srt")
    # predicted_subtiles = subs.Subtitles("./data/spider_man.srt")

    reference_subtitles = subs.Subtitles("./data/spider_man_source.srt", start=start, end=end)
    predicted_subtiles = subs.Subtitles("./output_subtitles/aligned_out.srt")

    accuracy = Accuracy()
    accuracy.calculate(reference_subtitles, predicted_subtiles)
    
    # reference_subtitles = subs.Subtitles("./data/spider_man_source.srt", start=start, end=end)
    # predicted_subtiles = subs.Subtitles("./data/spider_man_short_vocals.srt")

    # accuracy = Accuracy()
    # accuracy.calculate(reference_subtitles, predicted_subtiles)