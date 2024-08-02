
# Import necessary libraries
from asr.tokeniser import tokenize
import torch
import numpy as np

from unidecode import unidecode
from num2words import num2words
import numba
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import json
import nltk
from nltk.tokenize import sent_tokenize
import json

nltk.download('punkt')

# Function to convert time to seconds
def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second


# JIT-compiled function to process chunks of data efficiently
@numba.jit(nopython=True)
def process_chunk(scores, path_scores, row_start, row_end, num_rows, num_cols):
    # Define penalties for row and column movements
    row_penalty = 0.5
    col_penalty = 0.5

    # Loop through rows and columns to compute path scores
    for row in range(row_start, row_end):
        for col in range(1, scores.shape[1]):
            left = path_scores[row, col - 1]
            diag_left = path_scores[row - 1, col - 1]
            down = path_scores[row - 1, col]

            # Calculate the best previous score and apply penalties
            best_prev_score = max(left, diag_left, down)
            new_score = 0
            penalty = 0

            if best_prev_score == left:
                penalty = -row_penalty
                new_score = scores[row, col]
            elif best_prev_score == down:
                penalty = -col_penalty
                new_score = scores[row, col]
            elif best_prev_score == diag_left:
                penalty = -0
                new_score = scores[row, col]

            # Update the path scores with the calculated value
            path_scores[row, col] = max(best_prev_score + new_score + penalty, 0)


# Function to calculate path scores
def path_finder(scores, chunk_size=int(2**24), output_file="path_scores.dat"):
    # Create a memory-mapped array to store path scores
    path_scores = np.memmap(
        output_file, dtype=scores.dtype, mode="w+", shape=scores.shape
    )

    # Calculate the number of rows in each chunk
    num_rows_in_chunk = int(chunk_size / scores.shape[1])
    chunk_size = num_rows_in_chunk

    # Iterate through data in chunks and process each chunk
    for chunk_start in range(1, scores.shape[0] - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, scores.shape[0])
        chunk = scores[(chunk_start - 1) : chunk_end]
        chunk_path_scores = np.zeros(chunk.shape, dtype=scores.dtype)

        # Initialize the first row and column of the chunk's path scores
        if chunk_start == 1:
            chunk_path_scores[0, :] = chunk[0, :]
            chunk_path_scores[:, 0] = chunk[:, 0]
        else:
            chunk_path_scores[0, :] = path_scores[chunk_start - 1, :]
            chunk_path_scores[:, 0] = path_scores[chunk_start - 1 : chunk_end, 0]

        # Process the chunk to calculate path scores
        process_chunk(
            chunk,
            chunk_path_scores,
            1,
            chunk.shape[0],
            num_rows=scores.shape[0],
            num_cols=scores.shape[1],
        )

        # Update the overall path scores with the calculated chunk path scores
        path_scores[chunk_start - 1 : chunk_end] = chunk_path_scores

    # Normalize path scores and save them
    min_score = np.min(path_scores)
    max_score = np.max(path_scores)
    for i in range(path_scores.shape[0]):
        path_scores[i, :] = (path_scores[i, :] - min_score) / (max_score - min_score)
    path_scores.flush()
    del path_scores

    # Re-open the saved path scores for further use
    path_scores = np.memmap(
        output_file, dtype=scores.dtype, mode="r+", shape=scores.shape
    )
    return path_scores


# Function to reverse path and find optimal route
def reverse_path_finder(scores, path_scores):
    # Find the highest path score index
    idx = path_scores.argmax()
    row = int(idx / path_scores.shape[1])
    col = idx % path_scores.shape[1]

    path = []
    row_scores = [(row, col, scores[row, col])]
    col_scores = [(row, col, scores[row, col])]

    # Initialize dictionaries to store row and column scores
    row_dict = defaultdict(list)
    col_dict = defaultdict(list)

    # Traverse the path in reverse to find the optimal route
    while row > 0 and col > 0:
        left = path_scores[row, col - 1]
        diag_left = path_scores[row - 1, col - 1]
        down = path_scores[row - 1, col]

        best_prev_score = max(left, diag_left, down)

        # strout = f"({row}, {col}): left: {left}, diag: {diag_left}, down: {down} ----- "


        if best_prev_score == left:
            # strout += "Best direction: left"
            col -= 1

            if col_scores != []:
                path.append(col_scores)
                col_scores = []

            row_scores.append((row, col, scores[row, col]))

        elif best_prev_score == diag_left:
            # strout += "Best direction: diag"
            row -= 1
            col -= 1

            if col_scores != []:
                path.append(col_scores)
                col_scores = []

            if row_scores != []:
                path.append(row_scores)
                row_scores = []

            row_scores.append((row, col, scores[row, col]))
            col_scores.append((row, col, scores[row, col]))

        elif best_prev_score == down:
            # strout += "Best direction: down"
            row -= 1

            if row_scores != []:
                path.append(row_scores)
                row_scores = []

            col_scores.append((row, col, scores[row, col]))

        row_dict[row].append((row, col, scores[row, col]))
        col_dict[col].append((row, col, scores[row, col]))

        # print(strout)
    # Find optimal path by selecting the maximum value from each row and column
    new_path = []
    for row in path:
        max_idx = max(enumerate(row), key=lambda x: x[1][2])[0]
        new_path.append((row[max_idx][0], row[max_idx][1]))

    # Process row and column dictionaries to find optimal path
    for row, values in row_dict.items():
        max_idx = max(enumerate(values), key=lambda x: x[1][2])[0]
        row_dict[row] = (values[max_idx][0], values[max_idx][1])

    for col, values in col_dict.items():
        max_idx = max(enumerate(values), key=lambda x: x[1][2])[0]
        col_dict[col] = (values[max_idx][0], values[max_idx][1])

    # Combine row and column paths
    combined_path = list(row_dict.values()) + list(col_dict.values())
    # Remove duplicates by converting to a set and back to a list
    combined_path = list(set(combined_path))

    completed_rows = []
    path = []

    for i,j in combined_path:
        if i not in completed_rows:
            completed_rows.append(i)
            path.append((i, j))
    
    # Create a memory-mapped array to store reverse path scores
    reverse_path_scores = np.memmap(
        "reverse_path_finder.dat",
        dtype=path_scores.dtype,
        mode="w+",
        shape=path_scores.shape,
    )

    # Mark the optimal path in the reverse path scores array
    for idx in path:
        reverse_path_scores[idx] = 1
    reverse_path_scores.flush()
    
    # print(reverse_path_scores[sorted_path[0][0] + 1, sorted_path[0][1]])

    # Re-open the reverse path scores for further use
    reverse_path_scores = np.memmap(
        "reverse_path_finder.dat",
        dtype=path_scores.dtype,
        mode="r+",
        shape=path_scores.shape,
    )

    return reverse_path_scores



# Function to configure PyTorch settings
def configure_torch():
    torch.random.manual_seed(0)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return device


# Load tokens and create token-to-ID mapping
def load_tokens():
    tokens = [
        "-",
        "|",
        "e",
        "t",
        "a",
        "o",
        "n",
        "i",
        "h",
        "s",
        "r",
        "d",
        "l",
        "u",
        "m",
        "w",
        "c",
        "f",
        "g",
        "y",
        "p",
        "b",
        "v",
        "k",
        "'",
        "x",
        "j",
        "q",
        "z",
    ]
    tokens_to_id = {k: v for v, k in enumerate(tokens)}
    return tokens, tokens_to_id


# Load emission timesteps from a file
def load_emission_timesteps(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    predicted_timestamps = [float(number.replace("\n", "")) for number in lines[1:]]
    return predicted_timestamps


# Load emission data and process it
def load_emission_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    lines = [
        [
            float(number)
            for number in line.replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .split(",")
        ]
        for line in lines[1:]
    ]
    emission = torch.Tensor(lines)

    # Apply softmax to the emission data
    m = torch.nn.Softmax(dim=1)
    emission = m(emission)

    return emission


# Remove non-speech elements from queries
def remove_non_speech(queries):
    m = torch.nn.Softmax(dim=1)
    queries = m(queries)
    max_indices = torch.argmax(queries, dim=1)
    columns_to_remove = max_indices == 0
    return columns_to_remove

# Process subtitle data
def process_subtitles(subtitles, tokens):
    text = []
    timestamps = []
    for line in subtitles:
        start_time = line.start
        end_time = line.end
        line = line.text

        # Process subtitle text
        line = line.replace("\n", " ").replace("-", " ")
        line = " ".join([num2words(s) if s.isdigit() else s for s in line.split()])
        line = unidecode(line)
        line = [i if (i.isalpha() or i == " ") else "" for i in line]
        line = "".join(line).lower()

        # Extract tokens and timestamps
        for c in line:
            if c in tokens:
                text.append(c.lower())
                timestamps.append(
                    (
                        time_to_seconds(start_time.to_time()),
                        time_to_seconds(end_time.to_time()),
                    )
                )

            if c in [" ", "\n"]:
                text.append("|")
                timestamps.append(
                    (
                        time_to_seconds(start_time.to_time()),
                        time_to_seconds(end_time.to_time()),
                    )
                )

        text.append("|")
        timestamps.append(
            (time_to_seconds(start_time.to_time()), time_to_seconds(end_time.to_time()))
        )

    text = "".join(text).replace(" ", "|")
    return text, timestamps


# Calculate heatmap data
def calculate_heatmap_data(queries, keys):
    heatmap_data = queries.cpu()[:, keys.cpu()].numpy().T
    return heatmap_data


# Save heatmap image
def save_heatmap_image(heatmap_data, file_path):
    heatmap_data_im = (heatmap_data * 255).astype(np.uint8)[::-1]
    im = Image.fromarray(heatmap_data_im)
    im.save(file_path)

# Function to create word segments and their respective time intervals
def create_word_segments(char_time_pairs):
    words = []
    timesteps = []
    current_word = []
    current_times = []

    for char, (start, end) in char_time_pairs:
        if char != '|':
            current_word.append(char)
            if not current_times:
                current_times = [start, end]
            else:
                current_times[1] = end
        else:
            if current_word:
                words.append(''.join(current_word))
                timesteps.append({'start': current_times[0], 'end': current_times[1]})
                current_word = []
                current_times = []

    if current_word:
        words.append(''.join(current_word))
        timesteps.append({'start': current_times[0], 'end': current_times[1]})

    return words, timesteps


def create_sentence_timestep_mapping(original_chars, original_timesteps):
    # Join the character array into a single string
    original_text = ''.join(original_chars)
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(original_text)
    
    # Find the start and end indices of each sentence
    sentence_indices = []
    start_idx = 0
    for sentence in sentences:
        start_idx = original_text.find(sentence, start_idx)
        end_idx = start_idx + len(sentence) - 1
        sentence_indices.append((start_idx, end_idx))
        start_idx = end_idx + 1
    
    # Map sentence indices to timesteps
    sentence_timesteps = []
    for start_idx, end_idx in sentence_indices:
        start_timestep = original_timesteps[start_idx]
        end_timestep = original_timesteps[end_idx]
        sentence_timesteps.append({'start': start_timestep, 'end': end_timestep})
    
    # Create the result dictionary
    result = {
        'sentences': sentences,
        'timesteps': sentence_timesteps
    }
    
    return result
# Main function
def main(emission_file, emission_timesteps_file, input_text_file, output_json_file):
    device = configure_torch()

    tokens, tokens_to_id = load_tokens()
    predicted_timestamps = load_emission_timesteps(emission_timesteps_file)
    queries = load_emission_data(emission_file).to(device)

    non_speech_idx = remove_non_speech(queries)
    queries = queries[~non_speech_idx]

    predicted_timestamps = torch.Tensor(predicted_timestamps).to(device)
    predicted_timestamps = predicted_timestamps[~non_speech_idx]
    predicted_timestamps = predicted_timestamps.tolist()

    with open(input_text_file, 'r') as f:
        original_text = f.read()

    processed_text, index_map_original_to_processed, index_map_processed_to_original = tokenize(original_text)

    processed_text = "".join(processed_text).replace(" ", "|")
    tokens = [tokens_to_id[c] for c in processed_text]
    keys = torch.Tensor(tokens).type(torch.long).to(device)

    heatmap_data = calculate_heatmap_data(queries, keys)

    path_scores = path_finder(heatmap_data)
    heatmap_data_augmented = (path_scores * 255).astype(np.uint8)[::-1]

    reverse_path_scores = reverse_path_finder(heatmap_data, path_scores)
    reverse_heatmap_data_augmented = (reverse_path_scores * 255).astype(np.uint8)[::-1]

    reverse_path_scores_no_duplicates = reverse_path_scores
    reverse_heatmap_data_no_duplicates = (reverse_path_scores * 255).astype(np.uint8)[::-1]

    pred_text_indices, pred_timesteps = np.where(reverse_path_scores_no_duplicates == 1)

    original_chars = str(original_text)
    original_timesteps = [-1]*len(original_chars)

    for time_idx, processed_text_idx in zip(pred_timesteps, pred_text_indices):
        time = round(predicted_timestamps[time_idx], 2)
        original_text_idx = index_map_processed_to_original[processed_text_idx]
        orig_char = repr(original_text[original_text_idx])
        proc_char = repr(processed_text[processed_text_idx])

        original_timesteps[original_text_idx] = time

    for i, timestep in enumerate(original_timesteps):
        if timestep == -1:
            if i == 0:
                original_timesteps[i] = 0
            else:
                original_timesteps[i] = original_timesteps[i-1]

    # Generate the sentence-timestep mapping
    result = create_sentence_timestep_mapping(original_chars, original_timesteps)

    with open(output_json_file, "w") as json_file:
        json.dump(result, json_file, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process emission data and map it to text with timestamps.")
    parser.add_argument("--emission_file", type=str, required=True, help="File containing emission data")
    parser.add_argument("--emission_timesteps_file", type=str, required=True, help="File containing emission timesteps")
    parser.add_argument("--input_text_file", type=str, required=True, help="File containing the input text")
    parser.add_argument("--output_json_file", type=str, required=True, help="File to save the output JSON")

    args = parser.parse_args()

    main(args.emission_file, args.emission_timesteps_file, args.input_text_file, args.output_json_file)