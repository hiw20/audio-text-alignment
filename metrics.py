from sklearn.metrics import accuracy_score
import asr.subtitles as subs
import matplotlib.pyplot as plt
import numpy as np
import scipy
import plotly.graph_objects as go
import argparse
import plotly.graph_objects as go
import jiwer


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

        # Iterate over predicted texts and find matching times in reference texts
        for i in range(len(predicted_texts)):
            predicted_time = predicted_times[i]
            matching_times = self._find_matching_times(
                reference_times, predicted_time["start"], predicted_time["end"]
            )

            # Iterate over matching times and check if predicted text matches reference text
            for j in matching_times:
                # Correct prediction
                if predicted_texts[i] == reference_texts[j]:
                    pred_scores[i] = 1
                    break

        # Iterate over reference texts and find matching times in predicted texts
        for i in range(len(reference_texts)):
            reference_time = reference_times[i]
            matching_times = self._find_matching_times(
                predicted_times, reference_time["start"], reference_time["end"]
            )

            # Iterate over matching times and check if reference text matches predicted text
            for j in matching_times:
                # Correct reference
                if reference_texts[i] == predicted_texts[j]:
                    ref_scores[i] = 1
                    break

        # Calculate the number of correct and incorrect predictions
        correct_predictions = np.sum(pred_scores)
        incorrect_predictions = len(pred_scores) - correct_predictions

        # Calculate the number of correct and incorrect references
        correct_references = np.sum(ref_scores)
        incorrect_references = (
            max(len(ref_scores), len(pred_scores)) - correct_references
        )

        # Calculate the number of missed references
        missed = incorrect_references - incorrect_references

        # Calculate the total number of subtitles
        total_subtitles = max(len(predicted_texts), len(reference_texts))

        # Calculate the accuracy
        accuracy = correct_references / total_subtitles

        # Return the results
        return (
            correct_predictions,
            incorrect_predictions,
            incorrect_references,
            correct_references,
            missed,
            accuracy,
        )

    def _generate_accuracy_lists(self, reference_subtitles, predicted_subtitles):
        # Initializing an empty dictionary named 'ref_hash'
        ref_hash = {}

        # Initializing an empty dictionary named 'ref_enumerated'
        ref_enumerated = {}

        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            # Looping over the subtitles in 'reference_subtitles' and enumerating them

            ref_hash[i] = [ref_sub.start.to_time(), ref_sub.end.to_time()]
            # Storing the start and end times of each subtitle in 'ref_hash' dictionary using the enumeration index as the key

            ref_enumerated[ref_sub.text] = i
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
        # Looping over the subtitles in 'reference_subtitles' and enumerating them
        for i, ref_sub in enumerate(reference_subtitles.subtitles):
            # Appending the converted start time of the reference subtitle to 'ref_x' list
            ref_x.append(
                self._time_to_seconds(ref_hash[ref_enumerated[ref_sub.text]][0])
            )
            # Appending the enumeration index of the reference subtitle to 'ref_y' list
            ref_y.append(ref_enumerated[ref_sub.text])
        # Looping over the subtitles in 'predicted_subtitles'
        for pred_sub in predicted_subtitles.subtitles:
            # Checking if the text of the predicted subtitle exists in 'ref_enumerated'
            if pred_sub.text in ref_enumerated:
                # Appending the converted start time of the predicted subtitle to 'pred_x' list
                pred_x.append(self._time_to_seconds(pred_sub.start.to_time()))

                # Appending the enumeration index of the predicted subtitle to 'pred_y' list
                pred_y.append(ref_enumerated[pred_sub.text])

                # Retrieving the time period (start and end times) of the reference subtitle associated with the predicted subtitle
                time_period = ref_hash[ref_enumerated[pred_sub.text]]

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
        pred_enumerated = {}

        # Looping over the subtitles in 'predicted_subtitles' and enumerating them
        for i, pred_sub in enumerate(predicted_subtitles.subtitles):
            # Storing the start and end times of each subtitle in 'pred_hash' dictionary using the enumeration index as the key
            pred_hash[i] = [pred_sub.start.to_time(), pred_sub.end.to_time()]

            if pred_sub.text not in pred_enumerated:
                # Storing the text of each subtitle in 'pred_enumerated' dictionary with the enumeration index as the value
                pred_enumerated[pred_sub.text] = i

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
                < 3
                and abs(
                    self._time_to_seconds(time["end"]) - self._time_to_seconds(end_time)
                )
                < 3
            ):
                # Add the index to the list of matching times
                matching_times.append(i)

        # Return the list of matching times
        return matching_times


def get_results(filepath, filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end

    reference_subtitles = subs.Subtitles(
        "./data/" + filepath + filename + ".srt", start=start, end=end
    )
    decoded_subtitles = subs.Subtitles(
        "./output_subtitles/" + filename + "_decoded.srt"
    )

    # Subtitles aligned in real-time
    predicted_subtitles = subs.Subtitles(
        "./output_subtitles/" + filename + "_aligned_realtime.srt"
    )

    # Subtitles aligned once speech to text has run on entire audio. Should have higher accuracy.
    all_subtitles = subs.Subtitles(
        "./output_subtitles/" + filename + "_aligned_after.srt"
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
        " ".join(reference_subtitles.text),
        " ".join(decoded_subtitles.text).replace("|", " "),
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
    }


def plot_accuracy(
    data_values,
    title=None,
    x_title=None,
    y_title=None,
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
            )
        )
        if title:
            fig.update_layout(title=title, title_x=0.5)
        if x_title:
            fig.update_xaxes(title=x_title)
        if y_title:
            fig.update_yaxes(title=y_title, range=[0, 1])

    return fig


def plot_wer(
    data_values,
    title=None,
    x_title=None,
    y_title=None,
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


if __name__ == "__main__":
    clean_sample_results_dict = get_results(
        filepath="Samples/", filename="clean_sample"
    )

    print(f"Clean WER: {clean_sample_results_dict['word_error_rate']}")

    noise_levels = list(range(-29, -9))
    print(noise_levels)
    filepath = "Samples/"
    noise_filename = "clean_sample_noise_n{}db"
    sotu_filename = "clean_sample_sotu_n{}db"
    music_filename = "clean_sample_music_n{}db"

    noise_dictionaries = []
    noise_ratios = []
    sotu_dictionaries = []
    sotu_ratios = []
    music_dictionaries = []
    music_ratios = []

    for level in noise_levels:
        noise_results_dict = get_results(
            filepath=filepath, filename=noise_filename.format(abs(level))
        )
        noise_dictionaries.append(noise_results_dict)
        noise_ratios.append(level)

        sotu_results_dict = get_results(
            filepath=filepath, filename=sotu_filename.format(abs(level))
        )
        sotu_dictionaries.append(sotu_results_dict)
        sotu_ratios.append(level)

        music_results_dict = get_results(
            filepath=filepath, filename=music_filename.format(abs(level))
        )
        music_dictionaries.append(music_results_dict)
        music_ratios.append(level)

        # print(
        #     f"Noise{level} WER: {noise_results_dict['word_error_rate']}, Sotu{level} WER: {sotu_results_dict['word_error_rate']}, Music{level} WER: {music_results_dict['word_error_rate']}, "
        # )

        # wer.append(noise_results_dict["word_error_rate"])

        # # # Print the results of the accuracy calculation
        # # Accuracy._print_results(
        # #     clean_sample_noise_n1db_results_dict,
        # #     filename="clean_sample_noise_n1db",
        # # )
        # # # Plot the results using the generated coordinates
        # # Accuracy._plot_results(
        # #     clean_sample_noise_n1db_results_dict,
        # #     output_filename="clean_sample_noise_n1db",
        # # )

    data_values = zip(
        [noise_ratios, sotu_ratios, music_ratios],
        [noise_dictionaries, sotu_dictionaries, music_dictionaries],
        ["Pink Noise", "Speaking", "Music"],
    )
    fig = plot_accuracy(
        data_values,
        title="Accuracy vs Noise",
        x_title="RMS Noise Level (db)",
        y_title="Accuracy",
    )

    fig.add_hline(
        y=clean_sample_results_dict["accuracy"],
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="Sample accuracy without noise",
    )

    output_folder = "."
    output_filename = "noise_ratio_accuracy"
    fig.write_image(f"{output_folder}/{output_filename}.png")

    data_values = zip(
        [noise_ratios, sotu_ratios, music_ratios],
        [noise_dictionaries, sotu_dictionaries, music_dictionaries],
        ["Pink Noise", "Speaking", "Music"],
    )

    fig = plot_wer(
        data_values,
        title="WER vs Noise",
        x_title="RMS Noise Level (db)",
        y_title="WER",
    )

    fig.add_hline(
        y=clean_sample_results_dict["word_error_rate"],
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="Sample WER without noise",
    )

    output_folder = "."
    output_filename = "noise_ratio_wer"
    fig.write_image(f"{output_folder}/{output_filename}.png")
