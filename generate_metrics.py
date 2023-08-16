from sklearn.metrics import accuracy_score
import asr.subtitles as subs
import matplotlib.pyplot as plt
import numpy as np
import scipy
import plotly.graph_objects as go
import argparse
import plotly.graph_objects as go
import jiwer
from collections import defaultdict
from itertools import chain


# This is a class definition for SubtitleMetric.
# It is used to calculate a metric for comparing reference subtitles with predicted subtitles.


class SubtitleMetric:
    def __init__(self):
        pass
        # The __init__ method is a constructor for the SubtitleMetric class.
        # It initializes the class and does not perform any specific operations.

    def calculate(self, reference_subtitles, predicted_subtitles):
        raise NotImplementedError
        # The calculate method is a placeholder for calculating the metric.
        # It takes in two parameters: reference_subtitles and predicted_subtitles.
        # This method is intended to be overridden in the derived classes.
        # In this base class, it raises a NotImplementedError to indicate that it should be implemented in the derived classes.


class Accuracy(SubtitleMetric):
    # This class represents a metric for calculating accuracy of subtitle predictions

    def calculate(self, reference_subtitles, predicted_subtitles):
        # This method calculates the accuracy by comparing the reference subtitles with the predicted subtitles

        # Extract text and time data from reference subtitles
        ref_texts, ref_times = self._extract_subtitle_data(reference_subtitles)

        # Extract text and time data from predicted subtitles
        pred_texts, pred_times = self._extract_subtitle_data(predicted_subtitles)

        # Calculate various metrics based on the extracted subtitle data
        (
            correct_pred,  # Number of correctly predicted subtitles
            incorrect_pred,  # Number of incorrectly predicted subtitles
            incorrect_ref,  # Number of incorrectly referenced subtitles
            correct_ref,  # Number of correctly referenced subtitles
            missed,  # Number of missed subtitles
            accuracy,  # Accuracy percentage
        ) = self._calculate_accuracy(ref_texts, ref_times, pred_texts, pred_times)

        # Generate accuracy lists and retrieve the corresponding coordinates for plotting
        accuracy_list, ref_x, ref_y, pred_x, pred_y = self._generate_accuracy_lists(
            reference_subtitles, predicted_subtitles
        )

        # Calculate counts of correct, incorrect, and total subtitles
        correct, incorrect, total = self._calculate_counts(accuracy_list)

        # Return the results
        return (
            correct_pred,  # Number of correctly predicted subtitles
            incorrect_pred,  # Number of incorrectly predicted subtitles
            incorrect_ref,  # Number of incorrectly referenced subtitles
            correct_ref,  # Number of correctly referenced subtitles
            missed,  # Number of missed subtitles
            accuracy,  # Accuracy percentage
            accuracy_list,  # List of accuracy values for each subtitle
            ref_x,
            ref_y,
            pred_x,
            pred_y,
        )

    def _extract_subtitle_data(self, subtitles):
        # This method extracts text and time data from a given set of subtitles

        # Extract text from subtitles
        texts = [sub.text for sub in subtitles.subtitles]

        # Extract start and end times from subtitles
        times = [
            {"start": sub.start.to_time(), "end": sub.end.to_time()}
            for sub in subtitles.subtitles
        ]

        # Return the extracted text and time data
        return texts, times

    def _calculate_accuracy(
        self, reference_texts, reference_times, predicted_texts, predicted_times
    ):
        # Perform accuracy calculations and return the results

        # Initialize lists to store scores
        ref_scores = [0] * len(reference_texts)
        pred_scores = [0] * len(predicted_texts)

        ref_correct = []
        ref_incorrect = []
        ref_missed = []

        pred_correct = []
        pred_incorrect = []
        pred_missed = []

        # Iterate over predicted texts and find matching times in reference texts
        for i in range(len(predicted_texts)):
            predicted_time = predicted_times[i]
            matching_times = self._find_matching_times(
                reference_times, predicted_time["start"], predicted_time["end"]
            )

            match_found = False
            # Iterate over matching times and check if predicted text matches reference text
            for j in matching_times:
                # Correct prediction
                if predicted_texts[i] == reference_texts[j]:
                    pred_scores[i] = 1
                    match_found = True
                    break

            if len(matching_times) == 0:
                pred_missed.append(1)

            elif match_found:
                pred_correct.append(1)

            else:
                pred_incorrect.append(1)

        # Iterate over reference texts and find matching times in predicted texts
        for i in range(len(reference_texts)):
            reference_time = reference_times[i]
            matching_times = self._find_matching_times(
                predicted_times, reference_time["start"], reference_time["end"]
            )

            match_found = False
            # Iterate over matching times and check if reference text matches predicted text
            for j in matching_times:
                # Correct reference
                if reference_texts[i] == predicted_texts[j]:
                    ref_scores[i] = 1
                    match_found = True
                    break

            if len(matching_times) == 0:
                ref_missed.append(1)

            elif match_found:
                ref_correct.append(1)

            else:
                ref_incorrect.append(1)

        # Calculate the number of correct and incorrect predictions
        correct_predictions = np.sum(pred_scores)
        incorrect_predictions = len(pred_scores) - correct_predictions

        # Calculate the number of correct and incorrect references
        correct_references = np.sum(ref_scores)

        incorrect_references = len(ref_scores) - correct_references

        # Calculate the number of missed references
        missed = len(ref_scores) - correct_references

        # Calculate the total number of subtitles
        total_subtitles = max(len(predicted_texts), len(reference_texts))

        total_correct = max(sum(ref_correct), sum(pred_correct))
        total_incorrect = max(sum(ref_incorrect), sum(pred_incorrect))
        total_missed = sum(ref_missed) + sum(pred_missed)

        total_accuracy = total_correct / (
            total_correct + total_incorrect + total_missed
        )

        # Calculate the accuracy
        accuracy = correct_references / total_subtitles

        accuracy = total_accuracy

        # Return the results
        return (
            correct_predictions,
            incorrect_predictions,
            incorrect_references,
            correct_references,
            missed,
            accuracy,
        )

    def _create_subtitle_array(self, subtitle_texts, subtitle_times):
        if (
            len(subtitle_texts) < 1
            or len(subtitle_times) < 1
            or len(subtitle_texts) != len(subtitle_times)
        ):
            return []

        # Find the latest end time among all subtitles
        latest_end_time = max(t["end"] for t in subtitle_times)
        latest_end_time = self._time_to_seconds(latest_end_time)

        # Calculate the length of the array based on the latest end time
        array_length = int(latest_end_time) + 1

        # Create an array of -1s
        subtitle_array = [-1] * array_length

        # Populate the array with subtitle indices where applicable
        for i, text in enumerate(subtitle_texts):
            time = subtitle_times[i]
            start_time = self._time_to_seconds(time["start"])
            end_time = self._time_to_seconds(time["end"])

            # Convert start and end times to integer values
            start_index = int(start_time)
            end_index = int(end_time)

            # Populate the array within the start and end index range with the subtitle index
            for j in range(start_index, end_index + 1):
                subtitle_array[j] = i

        return subtitle_array

    def _generate_accuracy_lists(self, reference_subtitles, predicted_subtitles):
        # Initializing an empty dictionary named 'ref_hash'
        ref_hash = {}

        # Initializing an empty dictionary named 'ref_enumerated'
        ref_enumerated = defaultdict(list)  # {}

        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            # Looping over the subtitles in 'reference_subtitles' and enumerating them

            ref_hash[i] = [ref_sub.start.to_time(), ref_sub.end.to_time()]
            # Storing the start and end times of each subtitle in 'ref_hash' dictionary using the enumeration index as the key

            ref_enumerated[ref_sub.text].append(i)
        # Storing the text of each subtitle in 'ref_enumerated' dictionary with the enumeration index as the value

        # Initializing an empty list named 'accuracy_list'
        accuracy_list = []

        # Initializing an empty list named 'ref_x'
        ref_x = []

        # Initializing an empty list named 'ref_y'
        ref_y = []

        # Initializing an empty list named 'pred_x'
        pred_x = []

        # Initializing an empty list named 'pred_y'
        pred_y = []

        last_closest = 0
        # Looping over the subtitles in 'reference_subtitles' and enumerating them
        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            # Finding the closest key in 'ref_enumerated' to 'i' and appending its value to 'pred_y' list
            closest_key = min(
                ref_enumerated[ref_sub.text], key=lambda x: abs(x - last_closest)
            )
            last_closest = closest_key

            # print(
            #     f"I: {i}, index: {ref_enumerated[ref_sub.text]}, closest_key: {closest_key}"
            # )

            # Appending the converted start time of the reference subtitle to 'ref_x' list
            ref_x.append(
                self._time_to_seconds(ref_sub.start.to_time())
            )  # ref_hash[closest_key][0]))
            # Appending the enumeration index of the reference subtitle to 'ref_y' list
            ref_y.append(closest_key)

        last_closest = 0
        # Looping over the subtitles in 'predicted_subtitles'
        for i, pred_sub in enumerate(predicted_subtitles.subtitles):
            # Checking if the text of the predicted subtitle exists in 'ref_enumerated'
            if pred_sub.text in ref_enumerated:
                # Appending the converted start time of the predicted subtitle to 'pred_x' list
                pred_x.append(self._time_to_seconds(pred_sub.start.to_time()))

                # # Appending the enumeration index of the predicted subtitle to 'pred_y' list
                # pred_y.append(ref_enumerated[pred_sub.text])

                # Finding the closest key in 'ref_enumerated' to 'i' and appending its value to 'pred_y' list
                closest_key = min(
                    ref_enumerated[pred_sub.text], key=lambda x: abs(x - last_closest)
                )

                pred_y.append(closest_key)

                last_closest = closest_key

                # Retrieving the time period (start and end times) of the reference subtitle associated with the predicted subtitle
                time_period = ref_hash[closest_key]

                if (
                    self._time_to_seconds(time_period[0])
                    < self._time_to_seconds(pred_sub.start.to_time()) + 3
                    and self._time_to_seconds(pred_sub.end.to_time())
                    < self._time_to_seconds(time_period[1]) + 3
                ):
                    # If the predicted subtitle falls within a time range of +/- 3 seconds of the reference subtitle, append 1 to 'accuracy_list'
                    accuracy_list.append(1)
                else:
                    # If the predicted subtitle does not fall within the time range, append 0 to 'accuracy_list'
                    accuracy_list.append(0)
            else:
                # If the predicted subtitle does not exist in 'ref_enumerated', append 0 to 'accuracy_list'
                accuracy_list.append(0)

        # Initializing an empty dictionary named 'pred_hash'
        pred_hash = {}

        # Initializing an empty dictionary named 'pred_enumerated'
        pred_enumerated = defaultdict(list)  # {}

        # Looping over the subtitles in 'predicted_subtitles' and enumerating them
        for i, pred_sub in enumerate(predicted_subtitles.subtitles):
            # Storing the start and end times of each subtitle in 'pred_hash' dictionary using the enumeration index as the key
            pred_hash[i] = [pred_sub.start.to_time(), pred_sub.end.to_time()]

            if pred_sub.text not in pred_enumerated:
                # Storing the text of each subtitle in 'pred_enumerated' dictionary with the enumeration index as the value
                pred_enumerated[pred_sub.text].append(i)

        # Looping over the subtitles in 'reference_subtitles'
        for ref_sub in reference_subtitles.subtitles:
            if ref_sub.text not in pred_enumerated:
                # If the reference subtitle does not exist in 'pred_enumerated', append 0 to 'accuracy_list'
                accuracy_list.append(0)

        return accuracy_list, ref_x, ref_y, pred_x, pred_y

    # Function to calculate the counts of correct, incorrect, and total predictions
    def _calculate_counts(self, accuracy_list):
        # Summing up the elements in the accuracy_list to get the count of correct predictions
        correct = np.sum(accuracy_list)
        # Subtracting the count of correct predictions from the total count to get the count of incorrect predictions
        incorrect = len(accuracy_list) - correct
        # Total count of predictions
        total = len(accuracy_list)
        return correct, incorrect, total

    # Function to print the results
    def _print_results(results_dict, filename):
        # Printing the summary of prediction results
        print(
            "File: {}, Correct Predictions: {}, Incorrect Predictions: {}, Incorrect References: {}, Correct References: {}, Missed: {}, Accuracy: {}".format(
                filename,
                results_dict["correct_pred"],
                results_dict["incorrect_pred"],
                results_dict["incorrect_ref"],
                results_dict["correct_ref"],
                results_dict["missed"],
                results_dict["accuracy"],
            )
        )

    def _plot_results(results_dict, output_filename):
        ref_times = results_dict["ref_times"]
        ref_subtitle_indices = results_dict["ref_subtitle_indices"]
        pred_times = results_dict["pred_times"]
        pred_subtitle_indices = results_dict["pred_subtitle_indices"]

        # Plotting the reference data points in red and the predicted data points in blue
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ref_times,
                y=ref_subtitle_indices,
                mode="markers",
                marker=dict(color="red"),
                name="Reference",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pred_times,
                y=pred_subtitle_indices,
                mode="markers",
                marker=dict(color="blue"),
                name="Predicted",
            )
        )

        x0 = 0
        x1 = 1

        if "carnival_of_souls" in output_filename:
            x0 = 4260
            x1 = 4620
        elif "horror_express" in output_filename:
            x0 = 2136
            x1 = 2520
        elif "jungle_book" in output_filename:
            x0 = 1050
            x1 = 1250

        # Add vertical dashed lines at specified positions
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="red", opacity=0.2)

        # Update axis labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Subtitle Index")

        # Save the figure as an image
        fig.write_image(f"{output_filename}.png")

    def _generate_subtitle_comparison(
        self, ground_truth_subtitle, decoded_subtitle, aligned_subtitle
    ):
        # Open the file for comparison and clear its contents
        with open("./output_subtitles/compare.txt", "w") as file:
            pass  # Do nothing

        # Iterate through each subtitle in the ground truth subtitles
        for gt_sub in ground_truth_subtitle.subtitles:
            # Get the text of the current ground truth subtitle
            ground_truth_text = gt_sub.text

            # Get the decoded texts that fall within the time range of the current ground truth subtitle
            decoded_text = [
                dec_sub.text
                for dec_sub in decoded_subtitle.subtitles
                if self._is_within_time_range(gt_sub, dec_sub)
            ]

            # Get the aligned texts that fall within the time range of the current ground truth subtitle
            aligned_text = [
                ali_sub.text
                for ali_sub in aligned_subtitle.subtitles
                if self._is_within_time_range(gt_sub, ali_sub)
            ]

            # Open the comparison file in append mode
            with open("./output_subtitles/compare.txt", "a") as file:
                # Write the comparison information to the file
                file.write(
                    "Original\n{}\nDecoded\n{}\nAligned\n{}\n------------------------------------\n".format(
                        ground_truth_text, decoded_text, aligned_text
                    )
                )

    def _is_within_time_range(self, reference_sub, target_sub):
        # Get the start and end times of the reference and target subtitles in seconds
        reference_start_time = self._time_to_seconds(reference_sub.start.to_time())
        reference_end_time = self._time_to_seconds(reference_sub.end.to_time())
        target_start_time = self._time_to_seconds(target_sub.start.to_time())
        target_end_time = self._time_to_seconds(target_sub.end.to_time())

        # Check if the target subtitle falls within the time range of the reference subtitle
        return (reference_start_time <= target_start_time <= reference_end_time) or (
            reference_start_time <= target_end_time <= reference_end_time
        )

    def _time_to_seconds(self, time):
        # Convert the given time to seconds
        return time.hour * 3600 + time.minute * 60 + time.second

    def _find_matching_times(self, all_times, start_time, end_time):
        # Initialize a list to store the indices of matching times
        matching_times = []

        # Iterate through each time in the list of all times
        for i, time in enumerate(all_times):
            # Check if the start and end times are within a threshold of 3 seconds
            if (
                abs(
                    self._time_to_seconds(start_time)
                    - self._time_to_seconds(time["start"])
                )
                < 5
                and abs(
                    self._time_to_seconds(time["end"]) - self._time_to_seconds(end_time)
                )
                < 5
            ):
                # Add the index to the list of matching times
                matching_times.append(i)

        # Return the list of matching times
        return matching_times


def get_results(filepath, reference_filename, hypothesis_filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end

    reference_subtitles = subs.Subtitles(
        "./data/" + filepath + reference_filename + ".srt", start=start, end=end
    )
    decoded_subtitles = subs.Subtitles(
        "./output_subtitles/" + hypothesis_filename + "_decoded.srt"
    )

    # Subtitles aligned in real-time
    predicted_subtitles = subs.Subtitles(
        "./output_subtitles/" + hypothesis_filename + "_aligned_realtime.srt"
    )

    accuracy = Accuracy()
    (
        correct_pred,  # Number of correctly predicted subtitles
        incorrect_pred,  # Number of incorrectly predicted subtitles
        incorrect_ref,  # Number of incorrectly referenced subtitles
        correct_ref,  # Number of correctly referenced subtitles
        missed,  # Number of missed subtitles
        accuracy,  # Accuracy percentage
        accuracy_list,  # List of accuracy values for each subtitle
        ref_times,
        ref_subtitle_indices,
        pred_times,
        pred_subtitle_indices,
    ) = accuracy.calculate(reference_subtitles, predicted_subtitles)

    # print(" ".join(decoded_subtitles.text).replace("|", " "))
    word_error_rate = jiwer.wer(
        " ".join(reference_subtitles.whole_text),
        " ".join(decoded_subtitles.whole_text).replace("|", " "),
    )

    aligned_word_error_rate = jiwer.wer(
        " ".join(reference_subtitles.whole_text),
        " ".join(predicted_subtitles.whole_text),
    )

    return {
        "correct_pred": correct_pred,  # Number of correctly predicted subtitles
        "incorrect_pred": incorrect_pred,  # Number of incorrectly predicted subtitles
        "incorrect_ref": incorrect_ref,  # Number of incorrectly referenced subtitles
        "correct_ref": correct_ref,  # Number of correctly referenced subtitles
        "missed": missed,  # Number of missed subtitles
        "accuracy": accuracy,  # Accuracy percentage
        "accuracy_list": accuracy_list,  # List of accuracy values for each subtitle
        "ref_times": ref_times,
        "ref_subtitle_indices": ref_subtitle_indices,
        "pred_times": pred_times,
        "pred_subtitle_indices": pred_subtitle_indices,
        "word_error_rate": word_error_rate,
        "aligned_word_error_rate": aligned_word_error_rate,
    }


def plot_accuracy(
    data_values,
    title=None,
    x_title=None,
    y_title=None,
    line_colors=["red", "blue", "green", "red", "blue", "green"],
    line_dashes=["solid", "solid", "solid", "dash", "dash", "dash"],
):
    fig = go.Figure()
    for i, (x_values, dictionaries, plot_label) in enumerate(data_values):
        accuracy_values = [dictionary["accuracy"] for dictionary in dictionaries]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=accuracy_values,
                mode="lines",
                name=plot_label,
                legendrank=i,
                showlegend=True,
                line=dict(color=line_colors[i], dash=line_dashes[i]),
            )
        )
        if title:
            fig.update_layout(title=title, title_x=0.5)
        if x_title:
            fig.update_xaxes(title=x_title)
        if y_title:
            fig.update_yaxes(title=y_title)  # , range=[0, 1])

    return fig


def plot_wer(
    data_values,
    title=None,
    x_title=None,
    y_title=None,
    line_colors=["red", "blue", "green", "red", "blue", "green"],
    line_dashes=["solid", "solid", "solid", "dash", "dash", "dash"],
):
    fig = go.Figure()
    for i, (x_values, dictionaries, plot_label) in enumerate(data_values):
        wer_values = [dictionary["word_error_rate"] for dictionary in dictionaries]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=wer_values,
                mode="lines",
                name=plot_label,
                legendrank=i,
                showlegend=True,
                line=dict(color=line_colors[i], dash=line_dashes[i]),
            )
        )
        if title:
            fig.update_layout(title=title, title_x=0.5)
        if x_title:
            fig.update_xaxes(title=x_title)
        if y_title:
            fig.update_yaxes(
                title=y_title,
                rangemode="normal",
            )

    return fig


def run_for_single_models():
    file_types = ["noise", "speech", "music"]
    noise_levels = list(range(10, 30))
    negative_noise_levels = [-1 * level for level in noise_levels]

    vad_on = [True, False]
    model_names = [
        "WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H",
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_LARGE",
        "HUBERT_ASR_XLARGE",
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        # "whisper-large",
        "whisper-large-v2",
    ]

    for model_name in model_names:
        results = defaultdict(lambda: defaultdict(list))  # defaultdict(list)

        reference_filename = f"clean_sample"
        hypothesis_filename = f"{model_name}_vad_clean_sample"
        no_noise_vad_results_dict = get_results(
            filepath="Samples/",
            reference_filename=reference_filename,
            hypothesis_filename=hypothesis_filename,
        )

        # Plot the results using the generated coordinates
        Accuracy._plot_results(
            no_noise_vad_results_dict,
            output_filename=f"figures/{model_name}_vad_alignment",
        )

        hypothesis_filename = f"{model_name}_no_vad_clean_sample"
        no_noise_no_vad_results_dict = get_results(
            filepath="Samples/",
            reference_filename=reference_filename,
            hypothesis_filename=hypothesis_filename,
        )

        # Plot the results using the generated coordinates
        Accuracy._plot_results(
            no_noise_no_vad_results_dict,
            output_filename=f"figures/{model_name}_no_vad_alignment",
        )

        for vad in vad_on:
            vad = "vad" if vad else "no_vad"
            for file_type in file_types:
                for noise_level in noise_levels:
                    reference_filename = f"clean_sample_{file_type}_n{noise_level}db"
                    hypothesis_filename = (
                        f"{model_name}_{vad}_clean_sample_{file_type}_n{noise_level}db"
                    )
                    results_dict = get_results(
                        filepath="Samples/",
                        reference_filename=reference_filename,
                        hypothesis_filename=hypothesis_filename,
                    )
                    results[file_type][vad].append(results_dict)

        print(
            f"{model_name}:\
                    \n\tAccuracy vad {no_noise_vad_results_dict['accuracy']}:\
                    \tAccuracy no vad {no_noise_no_vad_results_dict['accuracy']}\
                    \n\tWER vad {no_noise_vad_results_dict['word_error_rate']}:\
                    \tWER no vad {no_noise_no_vad_results_dict['word_error_rate']}\
                    \n\tAligned WER vad {no_noise_vad_results_dict['aligned_word_error_rate']}:\
                    \tAligned WER no vad {no_noise_no_vad_results_dict['aligned_word_error_rate']}"
        )
        data_values = zip(
            [negative_noise_levels] * 6,
            [results[file_type]["vad"] for file_type in file_types]
            + [results[file_type]["no_vad"] for file_type in file_types],
            [f"{file_type}_vad" for file_type in file_types]
            + [f"{file_type}_no_vad" for file_type in file_types],
        )
        fig = plot_accuracy(
            data_values,
            title=f"{model_name} Accuracy vs Noise",
            x_title="RMS Noise Level (db)",
            y_title="Accuracy",
            line_colors=["red", "blue", "green", "red", "blue", "green"],
            line_dashes=["solid", "solid", "solid", "dot", "dot", "dot"],
        )

        fig.add_hline(
            y=no_noise_vad_results_dict["accuracy"],
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text="Sample accuracy no noise with vad",
        )

        fig.add_hline(
            y=no_noise_no_vad_results_dict["accuracy"],
            line_width=3,
            line_dash="dash",
            line_color="blue",
            annotation_text="Sample accuracy no noise no vad",
        )

        output_folder = "./figures"
        output_filename = f"{model_name}_noise_ratio_accuracy"
        fig.write_image(f"{output_folder}/{output_filename}.png")

        data_values = zip(
            [negative_noise_levels] * 6,
            [results[file_type]["vad"] for file_type in file_types]
            + [results[file_type]["no_vad"] for file_type in file_types],
            [f"{file_type}_vad" for file_type in file_types]
            + [f"{file_type}_no_vad" for file_type in file_types],
        )

        fig = plot_wer(
            data_values,
            title=f"{model_name} WER vs Noise",
            x_title="RMS Noise Level (db)",
            y_title="WER",
            line_colors=["red", "blue", "green", "red", "blue", "green"],
            line_dashes=["solid", "solid", "solid", "dot", "dot", "dot"],
        )

        fig.add_hline(
            y=no_noise_vad_results_dict["word_error_rate"],
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text="Sample WER no noise with vad",
        )

        fig.add_hline(
            y=no_noise_no_vad_results_dict["word_error_rate"],
            line_width=3,
            line_dash="dash",
            line_color="blue",
            annotation_text="Sample WER no noise no vad",
        )

        output_folder = "./figures"
        output_filename = f"{model_name}_noise_ratio_wer"
        fig.write_image(f"{output_folder}/{output_filename}.png")


def run_for_all_models_mean():
    file_types = ["noise", "speech", "music"]
    noise_levels = list(range(10, 30))
    negative_noise_levels = [-1 * level for level in noise_levels]

    vad_on = [True, False]
    model_names = [
        "WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H",
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_LARGE",
        "HUBERT_ASR_XLARGE",
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        # "whisper-large",
        "whisper-large-v2",
    ]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_name in model_names:
        for vad in vad_on:
            vad = "vad" if vad else "no_vad"

            for file_type in file_types:
                for noise_level in noise_levels:
                    reference_filename = f"clean_sample_{file_type}_n{noise_level}db"
                    hypothesis_filename = (
                        f"{model_name}_{vad}_clean_sample_{file_type}_n{noise_level}db"
                    )
                    results_dict = get_results(
                        filepath="Samples/",
                        reference_filename=reference_filename,
                        hypothesis_filename=hypothesis_filename,
                    )
                    results[model_name][vad][file_type].append(results_dict)

    for file_type in file_types:
        for vad in vad_on:
            vad = "vad" if vad else "no_vad"
            data_values = zip(
                [negative_noise_levels] * len(model_names),
                [results[model_name][vad][file_type] for model_name in model_names],
                [f"{model_name}_{vad}" for model_name in model_names],
            )

            fig = plot_accuracy(
                data_values,
                title=f"Accuracy vs Noise ({file_type})",
                x_title="RMS Noise Level (db)",
                y_title="Accuracy",
                line_colors=[
                    "hsl(0, 100%, 20%)",
                    "hsl(0, 100%, 50%)",
                    "hsl(0, 100%, 80%)",
                    "hsl(120, 100%, 30%)",
                    "hsl(120, 100%, 60%)",
                    "hsl(240, 100%, 20%)",
                    "hsl(240, 100%, 30%)",
                    "hsl(240, 100%, 50%)",
                    "hsl(240, 100%, 70%)",
                    "hsl(240, 100%, 90%)",
                ],
                line_dashes=[
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                ],
            )

            output_folder = "./figures"
            output_filename = f"mean_{file_type}_{vad}_accuracies"
            fig.write_image(f"{output_folder}/{output_filename}.png")


def run_for_long_audio():
    model_names = [
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_XLARGE",
        "whisper-medium",
    ]

    filenames = ["carnival_of_souls", "horror_express", "jungle_book"]

    for filename in filenames:
        for model_name in model_names:
            reference_filename = filename
            hypothesis_filename = f"{model_name}_no_vad_{filename}"
            print(f"Filename: {hypothesis_filename}")
            results_dict_no_vad = get_results(
                filepath="long_audio/",
                reference_filename=reference_filename,
                hypothesis_filename=hypothesis_filename,
            )

            # Plot the results using the generated coordinates
            Accuracy._plot_results(
                results_dict_no_vad,
                output_filename=f"figures/{model_name}_{filename}_no_vad_alignment",
            )

            # for model_name in model_names:
            reference_filename = filename
            hypothesis_filename = f"{model_name}_vad_{filename}"
            print(f"Filename: {hypothesis_filename}")
            results_dict_vad = get_results(
                filepath="long_audio/",
                reference_filename=reference_filename,
                hypothesis_filename=hypothesis_filename,
            )

            # Plot the results using the generated coordinates
            Accuracy._plot_results(
                results_dict_vad,
                output_filename=f"figures/{model_name}_{filename}_vad_alignment",
            )

            print(
                f"\n"
                f"{model_name}_{filename}"
                f"\n{'-' * 92}"
                f"\n{'':40}| {'VAD':^25} | {'No VAD':^25} |"
                f"\n{'-' * 92}"
                f"\n{'Accuracy':<40}| {results_dict_vad['accuracy']:<25} | {results_dict_no_vad['accuracy']:<25} |"
                f"\n{'-' * 92}"
                f"\n{'WER':<40}| {results_dict_vad['word_error_rate']:<25} | {results_dict_no_vad['word_error_rate']:<25} |"
                f"\n{'-' * 92}"
                f"\n{'Aligned WER':<40}| {results_dict_vad['aligned_word_error_rate']:<25} | {results_dict_no_vad['aligned_word_error_rate']:<25} |"
                f"\n{'-' * 92}"
                f"\n"
            )


def run_for_all_alignment_values():
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    vad_on = [True]
    model_names = [
        "WAV2VEC2_ASR_BASE_960H",
    ]

    modes = ["global", "local"]
    matchs = list(range(1, 11))
    end_gap_scores = [0, -1]

    for model_name in model_names:
        for vad in vad_on:
            for mode in modes:
                mode_info = f"_{mode}"
                for end_gap_score in end_gap_scores:
                    end_gap_info = f"_end_gap_{end_gap_score}"
                    for match in matchs:
                        match_info = f"_match_score_{match}"
                        extra_filename_info = mode_info + end_gap_info + match_info

                        for vad in vad_on:
                            vad = "vad" if vad else "no_vad"

                        reference_filename = f"clean_sample"
                        hypothesis_filename = (
                            f"{model_name}_{vad}_clean_sample_noise_n25db"
                            + extra_filename_info
                        )
                        results_dict = get_results(
                            filepath="Samples/",
                            reference_filename=reference_filename,
                            hypothesis_filename=hypothesis_filename,
                        )

                        end_gap = f"end_gap_{end_gap_score}"

                        results[model_name][mode][end_gap].append(results_dict)

    for model_name in model_names:
        for vad in vad_on:
            data_values_1 = []
            data_values_2 = []
            data_values_3 = []

            for mode in modes:
                mode_info = f"_{mode}"
                for end_gap_score in end_gap_scores:
                    vad = "vad" if vad else "no_vad"
                    end_gap = f"end_gap_{end_gap_score}"

                    data_values_1.extend([matchs] * len(model_names))
                    data_values_2.extend(
                        [
                            results[model_name][mode][end_gap]
                            for model_name in model_names
                        ]
                    )
                    data_values_3.extend(
                        [f"{model_name}_{mode}_{end_gap}" for model_name in model_names]
                    )

            data_values = zip(data_values_1, data_values_2, data_values_3)

            fig = plot_accuracy(
                data_values,
                title=f"Accuracy vs Alignment Parameters",
                x_title="Match Score",
                y_title="Accuracy",
                line_colors=[
                    "hsl(0, 100%, 20%)",
                    "hsl(0, 100%, 50%)",
                    "hsl(0, 100%, 80%)",
                    "hsl(120, 100%, 30%)",
                    "hsl(120, 100%, 60%)",
                    "hsl(240, 100%, 20%)",
                    "hsl(240, 100%, 30%)",
                    "hsl(240, 100%, 50%)",
                    "hsl(240, 100%, 70%)",
                    "hsl(240, 100%, 90%)",
                ],
                line_dashes=[
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                    "solid",
                ],
            )

            output_folder = "./figures"
            output_filename = f"alignment_accuracies"
            fig.write_image(f"{output_folder}/{output_filename}.png")


if __name__ == "__main__":
    run_for_single_models()
    run_for_all_models_mean()
    run_for_long_audio()
    run_for_all_alignment_values()
