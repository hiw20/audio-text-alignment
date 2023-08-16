# import torch
# import numpy as np
# import asr.subtitles as subs
# from unidecode import unidecode
# from num2words import num2words
# import numba
# from PIL import Image
# from collections import defaultdict


# def time_to_seconds(time):
#     return time.hour * 3600 + time.minute * 60 + time.second


# @numba.jit(nopython=True)
# def process_chunk(scores, path_scores, row_start, row_end, num_rows, num_cols):
#     row_penalty = 0.5
#     col_penalty = 0.5

#     for row in range(row_start, row_end):
#         for col in range(1, scores.shape[1]):
#             left = path_scores[row, col - 1]
#             diag_left = path_scores[row - 1, col - 1]
#             down = path_scores[row - 1, col]

#             best_prev_score = max(left, diag_left, down)
#             new_score = 0
#             penalty = 0

#             if best_prev_score == left:
#                 penalty = -row_penalty
#                 new_score = scores[row, col]
#             elif best_prev_score == down:
#                 penalty = -col_penalty
#                 new_score = scores[row, col]
#             elif best_prev_score == diag_left:
#                 penalty = -0
#                 new_score = scores[row, col]

#             path_scores[row, col] = max(best_prev_score + new_score + penalty, 0)


# def path_finder(scores, chunk_size=int(2**24), output_file="path_scores.dat"):
#     path_scores = np.memmap(
#         output_file, dtype=scores.dtype, mode="w+", shape=scores.shape
#     )
#     num_rows_in_chunk = int(chunk_size / scores.shape[1])
#     chunk_size = num_rows_in_chunk

#     for chunk_start in range(1, scores.shape[0] - 1, chunk_size):
#         chunk_end = min(chunk_start + chunk_size, scores.shape[0])
#         chunk = scores[(chunk_start - 1) : chunk_end]
#         chunk_path_scores = np.zeros(chunk.shape, dtype=scores.dtype)

#         if chunk_start == 1:
#             chunk_path_scores[0, :] = chunk[0, :]
#             chunk_path_scores[:, 0] = chunk[:, 0]
#         else:
#             chunk_path_scores[0, :] = path_scores[chunk_start - 1, :]
#             chunk_path_scores[:, 0] = path_scores[chunk_start - 1 : chunk_end, 0]

#         process_chunk(
#             chunk,
#             chunk_path_scores,
#             1,
#             chunk.shape[0],
#             num_rows=scores.shape[0],
#             num_cols=scores.shape[1],
#         )

#         path_scores[chunk_start - 1 : chunk_end] = chunk_path_scores

#     min_score = np.min(path_scores)
#     max_score = np.max(path_scores)

#     for i in range(path_scores.shape[0]):
#         path_scores[i, :] = (path_scores[i, :] - min_score) / (max_score - min_score)

#     path_scores.flush()
#     del path_scores

#     path_scores = np.memmap(
#         output_file, dtype=scores.dtype, mode="r+", shape=scores.shape
#     )
#     return path_scores


# def reverse_path_finder(scores, path_scores):
#     idx = path_scores.argmax()
#     row = int(idx / path_scores.shape[1])
#     col = idx % path_scores.shape[1]

#     path = []

#     row_scores = [(row, col, scores[row, col])]
#     col_scores = [(row, col, scores[row, col])]

#     row_dict = defaultdict(list)
#     col_dict = defaultdict(list)

#     while row > 0 and col > 0:
#         left = path_scores[row, col - 1]
#         diag_left = path_scores[row - 1, col - 1]
#         down = path_scores[row - 1, col]

#         best_prev_score = max(left, diag_left, down)

#         if best_prev_score == left:
#             col -= 1

#             if col_scores != []:
#                 path.append(col_scores)
#                 col_scores = []

#             row_scores.append((row, col, scores[row, col]))

#         elif best_prev_score == diag_left:
#             row -= 1
#             col -= 1

#             if col_scores != []:
#                 path.append(col_scores)
#                 col_scores = []

#             if row_scores != []:
#                 path.append(row_scores)
#                 row_scores = []

#             row_scores.append((row, col, scores[row, col]))
#             col_scores.append((row, col, scores[row, col]))

#         elif best_prev_score == down:
#             row -= 1

#             if row_scores != []:
#                 path.append(row_scores)
#                 row_scores = []

#             col_scores.append((row, col, scores[row, col]))

#         row_dict[row].append((row, col, scores[row, col]))
#         col_dict[col].append((row, col, scores[row, col]))

#     new_path = []
#     for row in path:
#         max_idx = max(enumerate(row), key=lambda x: x[1][2])[0]

#         new_path.append((row[max_idx][0], row[max_idx][1]))

#     path = new_path

#     for row, values in row_dict.items():
#         print(f"(row: {row}): {values}")
#         max_idx = max(enumerate(values), key=lambda x: x[1][2])[0]
#         row_dict[row] = (values[max_idx][0], values[max_idx][1])

#     for col, values in col_dict.items():
#         print(f"(col: {col}): {values}")
#         max_idx = max(enumerate(values), key=lambda x: x[1][2])[0]
#         col_dict[col] = (values[max_idx][0], values[max_idx][1])

#     path = set(row_dict.values()).intersection(set(col_dict.values()))
#     path = list(path)

#     reverse_path_scores = np.memmap(
#         "reverse_path_finder.dat",
#         dtype=path_scores.dtype,
#         mode="w+",
#         shape=path_scores.shape,
#     )

#     for idx in path:
#         reverse_path_scores[idx] = 1

#     reverse_path_scores.flush()

#     reverse_path_scores = np.memmap(
#         "reverse_path_finder.dat",
#         dtype=path_scores.dtype,
#         mode="r+",
#         shape=path_scores.shape,
#     )

#     return reverse_path_scores


# def configure_torch():
#     torch.random.manual_seed(0)
#     device = torch.device("cpu")
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return device

# def load_tokens():
#     tokens = [
#         "-",
#         "|",
#         "e",
#         "t",
#         "a",
#         "o",
#         "n",
#         "i",
#         "h",
#         "s",
#         "r",
#         "d",
#         "l",
#         "u",
#         "m",
#         "w",
#         "c",
#         "f",
#         "g",
#         "y",
#         "p",
#         "b",
#         "v",
#         "k",
#         "'",
#         "x",
#         "j",
#         "q",
#         "z",
#     ]
#     tokens_to_id = {k: v for v, k in enumerate(tokens)}
#     return tokens, tokens_to_id


# def load_emission_timesteps(file_path):
#     with open(file_path, "r") as file:
#         lines = file.readlines()
#     predicted_timestamps = [float(number.replace("\n", "")) for number in lines[1:]]
#     return predicted_timestamps


# def load_emission_data(file_path):
#     with open(file_path, "r") as file:
#         lines = file.readlines()
#     lines = [
#         [
#             float(number)
#             for number in line.replace("\n", "")
#             .replace("[", "")
#             .replace("]", "")
#             .replace(" ", "")
#             .split(",")
#         ]
#         for line in lines[1:]
#     ]
#     emission = torch.Tensor(lines)

#     # Process data to obtain queries, predicted_timestamps, etc.
#     m = torch.nn.Softmax(dim=1)
#     emission = m(emission)

#     return emission


# def remove_non_speech(queries):
#     m = torch.nn.Softmax(dim=1)
#     queries = m(queries)
#     max_indices = torch.argmax(queries, dim=1)
#     columns_to_remove = max_indices == 0
#     return columns_to_remove


# def process_subtitles(subtitles, tokens):
#     text = []
#     timestamps = []
#     for line in subtitles:
#         start_time = line.start
#         end_time = line.end
#         line = line.text

#         line = line.replace("\n", " ").replace("-", " ")
#         line = " ".join([num2words(s) if s.isdigit() else s for s in line.split()])
#         line = unidecode(line)
#         line = [i if (i.isalpha() or i == " ") else "" for i in line]
#         line = "".join(line).lower()

#         for c in line:
#             if c in tokens:
#                 text.append(c.lower())
#                 timestamps.append(
#                     (
#                         time_to_seconds(start_time.to_time()),
#                         time_to_seconds(end_time.to_time()),
#                     )
#                 )

#             if c in [" ", "\n"]:
#                 text.append("|")
#                 timestamps.append(
#                     (
#                         time_to_seconds(start_time.to_time()),
#                         time_to_seconds(end_time.to_time()),
#                     )
#                 )

#         text.append("|")
#         timestamps.append(
#             (time_to_seconds(start_time.to_time()), time_to_seconds(end_time.to_time()))
#         )

#     text = "".join(text).replace(" ", "|")
#     return text, timestamps


# def calculate_heatmap_data(queries, keys):
#     heatmap_data = queries.cpu()[:, keys.cpu()].numpy().T
#     return heatmap_data


# def save_heatmap_image(heatmap_data, file_path):
#     heatmap_data_im = (heatmap_data * 255).astype(np.uint8)[::-1]
#     im = Image.fromarray(heatmap_data_im)
#     im.save(file_path)


# def main():
#     device = configure_torch()

#     tokens, tokens_to_id = load_tokens()
#     predicted_timestamps = load_emission_timesteps("emission_timesteps.txt")
#     queries = load_emission_data("emission.txt").to(device)

#     non_speech_idx = remove_non_speech(queries)

#     queries = queries[~non_speech_idx]

#     predicted_timestamps = torch.Tensor(predicted_timestamps).to(device)
#     predicted_timestamps = predicted_timestamps[~non_speech_idx]
#     predicted_timestamps = predicted_timestamps.tolist()

#     reference_subtitles = subs.Subtitles(
#         "./data/long_audio/jungle_book.srt", start=0, end=50
#     )
#     subtitles = reference_subtitles.subtitles
#     text, timestamps = process_subtitles(subtitles, tokens)

#     text = "".join(text).replace(" ", "|")
#     tokens = [tokens_to_id[c] for c in text]
#     keys = torch.Tensor(tokens).type(torch.long).to(device)

#     heatmap_data = calculate_heatmap_data(queries, keys)
#     save_heatmap_image(heatmap_data, "heatmap.png")

#     path_scores = path_finder(heatmap_data)
#     heatmap_data_augmented = (path_scores * 255).astype(np.uint8)[::-1]
#     im = Image.fromarray(heatmap_data_augmented, "L")
#     im.save("figures/heatmap_processed.png")

#     reverse_path_scores = reverse_path_finder(heatmap_data, path_scores)
#     reverse_heatmap_data_augmented = (reverse_path_scores * 255).astype(np.uint8)[::-1]
#     im = Image.fromarray(reverse_heatmap_data_augmented, "L")
#     im.save("figures/heatmap_processed_reverse.png")

#     reverse_path_scores_no_duplicates = reverse_path_scores
#     reverse_heatmap_data_no_duplicates = (reverse_path_scores * 255).astype(np.uint8)[
#         ::-1
#     ]
#     heatmap_red = Image.fromarray(reverse_heatmap_data_no_duplicates, mode="L")

#     # Apply red color to heatmap_red where the value is 255
#     heatmap_red_colored = Image.new("RGB", heatmap_red.size)
#     red_overlay = Image.new("RGB", heatmap_red.size, color=(255, 0, 0))
#     heatmap_red_colored.paste(red_overlay, mask=heatmap_red)

#     heatmap_data_im = (heatmap_data * 255).astype(np.uint8)[::-1]
#     heatmap_bw = Image.fromarray(heatmap_data_im, mode="L")
#     heatmap_bw = heatmap_bw.convert("RGB")

#     # Overlay the images using the "blend" function (you can use other blending modes as well)
#     overlay = Image.blend(heatmap_red_colored, heatmap_bw, alpha=0.5)
#     overlay.save("figures/heatmap_overlay.png")

#     pred_text_indices, pred_timesteps = np.where(reverse_path_scores_no_duplicates == 1)
#     accuracy = []
#     missed = []
#     start_time = 203  # Adjust as needed

#     for pred_text_idx, pred_timestamp_idx in zip(pred_text_indices, pred_timesteps):
#         if text[pred_text_idx] != "|":
#             start_condition = timestamps[pred_text_idx][0] - 5
#             end_condition = timestamps[pred_text_idx][1] + 5
#             predicted_time = predicted_timestamps[pred_timestamp_idx] + start_time

#             if start_condition <= predicted_time <= end_condition:
#                 accuracy.append(1)
#             else:
#                 accuracy.append(0)
#                 missed.append(
#                     (
#                         text[pred_text_idx],
#                         timestamps[pred_text_idx],
#                         predicted_time,
#                     )
#                 )

#     print(f"Missed: {missed}")
#     print(f"Accuracy: {np.sum(accuracy) / len(accuracy)}")


# if __name__ == "__main__":
#     main()


# Import necessary libraries
import torch
import numpy as np
import asr.subtitles as subs
from unidecode import unidecode
from num2words import num2words
import numba
from PIL import Image
from collections import defaultdict


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

        if best_prev_score == left:
            col -= 1

            if col_scores != []:
                path.append(col_scores)
                col_scores = []

            row_scores.append((row, col, scores[row, col]))

        elif best_prev_score == diag_left:
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
            row -= 1

            if row_scores != []:
                path.append(row_scores)
                row_scores = []

            col_scores.append((row, col, scores[row, col]))

        row_dict[row].append((row, col, scores[row, col]))
        col_dict[col].append((row, col, scores[row, col]))

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

    # Find common elements in row and column paths
    path = set(row_dict.values()).intersection(set(col_dict.values()))
    path = list(path)

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


# Main function
def main():
    device = configure_torch()

    tokens, tokens_to_id = load_tokens()
    predicted_timestamps = load_emission_timesteps("emission_timesteps.txt")
    queries = load_emission_data("emission.txt").to(device)

    non_speech_idx = remove_non_speech(queries)
    queries = queries[~non_speech_idx]

    predicted_timestamps = torch.Tensor(predicted_timestamps).to(device)
    predicted_timestamps = predicted_timestamps[~non_speech_idx]
    predicted_timestamps = predicted_timestamps.tolist()

    reference_subtitles = subs.Subtitles(
        "./data/long_audio/jungle_book.srt", start=0, end=50
    )
    subtitles = reference_subtitles.subtitles
    text, timestamps = process_subtitles(subtitles, tokens)

    text = "".join(text).replace(" ", "|")
    tokens = [tokens_to_id[c] for c in text]
    keys = torch.Tensor(tokens).type(torch.long).to(device)

    heatmap_data = calculate_heatmap_data(queries, keys)
    save_heatmap_image(heatmap_data, "heatmap.png")

    path_scores = path_finder(heatmap_data)
    heatmap_data_augmented = (path_scores * 255).astype(np.uint8)[::-1]
    im = Image.fromarray(heatmap_data_augmented, "L")
    im.save("figures/heatmap_processed.png")

    reverse_path_scores = reverse_path_finder(heatmap_data, path_scores)
    reverse_heatmap_data_augmented = (reverse_path_scores * 255).astype(np.uint8)[::-1]
    im = Image.fromarray(reverse_heatmap_data_augmented, "L")
    im.save("figures/heatmap_processed_reverse.png")

    reverse_path_scores_no_duplicates = reverse_path_scores
    reverse_heatmap_data_no_duplicates = (reverse_path_scores * 255).astype(np.uint8)[
        ::-1
    ]
    heatmap_red = Image.fromarray(reverse_heatmap_data_no_duplicates, mode="L")

    # Create a red heatmap overlay
    heatmap_red_colored = Image.new("RGB", heatmap_red.size)
    red_overlay = Image.new("RGB", heatmap_red.size, color=(255, 0, 0))
    heatmap_red_colored.paste(red_overlay, mask=heatmap_red)

    heatmap_data_im = (heatmap_data * 255).astype(np.uint8)[::-1]
    heatmap_bw = Image.fromarray(heatmap_data_im, mode="L")
    heatmap_bw = heatmap_bw.convert("RGB")

    # Overlay the images using a blend function
    overlay = Image.blend(heatmap_red_colored, heatmap_bw, alpha=0.5)
    overlay.save("figures/heatmap_overlay.png")

    # Calculate accuracy and missed predictions
    pred_text_indices, pred_timesteps = np.where(reverse_path_scores_no_duplicates == 1)
    accuracy = []
    missed = []
    start_time = 203  # Adjust as needed

    for pred_text_idx, pred_timestamp_idx in zip(pred_text_indices, pred_timesteps):
        if text[pred_text_idx] != "|":
            start_condition = timestamps[pred_text_idx][0] - 5
            end_condition = timestamps[pred_text_idx][1] + 5
            predicted_time = predicted_timestamps[pred_timestamp_idx] + start_time

            if start_condition <= predicted_time <= end_condition:
                accuracy.append(1)
            else:
                accuracy.append(0)
                missed.append(
                    (
                        text[pred_text_idx],
                        timestamps[pred_text_idx],
                        predicted_time,
                    )
                )

    # Print results
    print(f"Missed: {missed}")
    print(f"Accuracy: {np.sum(accuracy) / len(accuracy)}")


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
