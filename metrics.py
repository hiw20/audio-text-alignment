from sklearn.metrics import accuracy_score
from asr.aligner import needleman_wunsch
import asr.subtitles as subs
import matplotlib.pyplot as plt
import numpy as np
import scipy

class Metric:
    def __init__(self):
        pass

    def calculate(self, reference_subtitles, predicted_subtitles):
        raise NotImplementedError

def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second

def find_time(times, start_time, end_time):
    correct_times = []

    for i, time in enumerate(times):
        if abs(time_to_seconds(start_time) - time_to_seconds(time["start"])) < 3 and \
            abs(time_to_seconds(time["end"]) - time_to_seconds(end_time)) < 3:
                correct_times.append(i)

    return correct_times

def correct_check(reference, predicted):
    ref_text, ref_time, ref_score = [], [], []
    pred_text, pred_time, pred_score = [], [], []

    for subtitle in reference.subtitles:
        ref_text.append(subtitle.text)
        ref_time.append({'start': subtitle.start.to_time(), 'end': subtitle.end.to_time()})
        ref_score.append(0)
    
    for subtitle in predicted.subtitles:
        pred_text.append(subtitle.text)
        pred_time.append({'start': subtitle.start.to_time(), 'end': subtitle.end.to_time()})
        pred_score.append(0)
    
    for i in range(len(pred_text)):
        predicted_time = pred_time[i]
        correct_times = find_time(ref_time, predicted_time["start"], predicted_time["end"])


        for j in correct_times:
            #Correct
            if pred_text[i] == ref_text[j]:
                pred_score[i] = 1
                break
    
    for i in range(len(ref_text)):
        reference_time = ref_time[i]
        correct_times = find_time(pred_time, reference_time["start"], reference_time["end"])


        for j in correct_times:
            #Correct
            if ref_text[i] == pred_text[j]:
                ref_score[i] = 1
                break
    
    correct = np.sum(pred_score)
    incorrect = len(pred_score) - correct

    correct_ref = np.sum(ref_score)
    incorrect_ref = max(len(ref_score), len(pred_score)) - correct_ref

    missed = incorrect_ref - incorrect_ref

    total = max(len(pred_text), len(ref_text))
    accuracy = correct_ref / total

    print("Correct: {}, Incorrect: {}, Incorrect Ref: {}, Correct Ref: {}, Missed: {}, Accuracy: {}".format(correct, incorrect, incorrect_ref, correct_ref, missed, accuracy))
    
    # from collections import defaultdict
    # import asr.needleman_wunsch as nw

    # sub_hash = defaultdict(lambda:-1)
    # for i, text in enumerate(ref_text[::-1] + pred_text[::-1]):
    #     if text not in sub_hash:
    #         sub_hash[text] = i

    # sub_hash_inv = defaultdict(lambda:-1)

    # for k, v in sub_hash.items():
    #     sub_hash_inv[v] = k

    # pred_hash = [sub_hash[word] for word in pred_text]
    # ref_hash = [sub_hash[word] for word in ref_text]

    # x, y = nw.nw(pred_hash, ref_hash, gap=1, mismatch=1, match=1)
    # x = [i if i != -1 else None for i in x]
    # y = [i if i != -1 else None for i in y]

    # pred_align = [sub_hash_inv[i] if i is not None else "-" for i in x]
    # ref_align = [sub_hash_inv[i] if i is not None else "-" for i in y]

    # correct = 0
    # incorrect_pred = 0
    # incorrect_ref = 0

    # for i in range(len(pred_align)):
    #     p = pred_align[i]
    #     r = ref_align[i]
    #     if p == r:
    #         correct += 1
    #     if p == "-":
    #         incorrect_pred += 1
    #     if r == "-":
    #         correct_pred += 1

    # print("Correct: {}, Incorrect Pred: {}, Incorrect Ref: {}".format(correct, incorrect_pred, incorrect_ref))


    # with open("output_subtitles/metrics.txt", 'w') as f:
    #     f.write("")

    # with open("output_subtitles/metrics.txt", 'a') as f:
    #     for i,j in zip(pred_align,ref_align):
    #         i = i.replace("\n"," ")
    #         j = j.replace("\n"," ")
    #         f.write("{}\t\t\t{}\n".format(i[:min(10, len(i))], j[:min(10, len(j))]))





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
                    # print("Incorrect: {}\t{}\t{}".format(pred_sub.start, ref_enumerated[pred_sub.text], pred_sub.text))
            else:
                accuracy.append(0)

        correct = sum(accuracy)
        incorrect = len(accuracy) - correct

        pred_hash = {}
        pred_enumerated = {}
        for i, pred_sub in enumerate(predicted_subtitles.subtitles):
            pred_hash[i] = [pred_sub.start.to_time(), pred_sub.end.to_time()]
            if pred_sub.text not in pred_enumerated:
                pred_enumerated[pred_sub.text] = i

        for ref_sub in reference_subtitles.subtitles:
            if ref_sub.text not in pred_enumerated:
                accuracy.append(0)
                # print("Missed: {}\t{}\t{}".format(ref_sub.start, ref_enumerated[ref_sub.text], ref_sub.text))
        
        total = len(accuracy)
        
        print("Correct: {}, Incorrect: {}, Missed: {}, Total: {}, Accuracy: {}".format(correct, incorrect, len(accuracy) - correct - incorrect, total, correct/total))

        # plt.plot(range(len(ref_x)), ref_y, 'r.')
        # plt.plot(range(len(pred_x)), pred_y, 'b.')

        plt.plot(ref_x, ref_y, 'r.')
        plt.plot(pred_x, pred_y, 'b.')
        # plt.plot(pred_x, np.convolve(pred_y, np.ones(10)/10, 'same'), 'g.')
        #Save figure
        plt.savefig('plot.png')

def get_seconds(time):
    return time.hours*3600.0 + time.minutes*60.0 + time.seconds

def align(ground_truth, decoded, aligned):
    with open("./output_subtitles/compare.txt", "w") as f:
        pass
    
    for gt_s in ground_truth.subtitles:
        g = gt_s.text
        d = []
        a = []

        for d_s in decoded.subtitles:
            if get_seconds(gt_s.start) <= get_seconds(d_s.start) <= get_seconds(gt_s.end)\
                or get_seconds(gt_s.start) <= get_seconds(d_s.end) <= get_seconds(gt_s.end):
                
                d.append(d_s.text)
        
        for a_s in aligned.subtitles:
            if get_seconds(gt_s.start) <= get_seconds(a_s.start) <= get_seconds(gt_s.end)\
                or get_seconds(gt_s.start) <= get_seconds(a_s.end) <= get_seconds(gt_s.end):
                
                a.append(a_s.text)
        
        with open("./output_subtitles/compare.txt", "a") as f:
            f.write("Original\n{}\nDecoded\n{}\nAligned\n{}\n------------------------------------\n".format(g, d, a))
        # print("{}\t{}".format(g, d))

    

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
    decoded_subtiles = subs.Subtitles("./output_subtitles/decoder_out.srt")
    predicted_subtiles = subs.Subtitles("./output_subtitles/aligned_out.srt")
    all_subtiles = subs.Subtitles("./output_subtitles/all_out.srt")

    accuracy = Accuracy()
    accuracy.calculate(reference_subtitles, predicted_subtiles)
    # accuracy.calculate(reference_subtitles, all_subtiles)

    align(reference_subtitles, decoded_subtiles, predicted_subtiles)

    correct_check(reference_subtitles, predicted_subtiles)
    correct_check(reference_subtitles, all_subtiles)
    
    # reference_subtitles = subs.Subtitles("./data/spider_man_source.srt", start=start, end=end)
    # predicted_subtiles = subs.Subtitles("./data/spider_man_short_vocals.srt")

    # accuracy = Accuracy()
    # accuracy.calculate(reference_subtitles, predicted_subtiles)