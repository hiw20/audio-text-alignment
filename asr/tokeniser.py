from unidecode import unidecode
from num2words import num2words
import re

def tokenize(text):
    """
    Tokenizes text by removing diacritics, replacing digits with words, and keeping only 
    alphabetic characters or spaces. Words are separated by "|" characters.
    Also returns a list of tuples (original_index, new_index).
    """

    # Intermediate step to store text and mapping
    intermediate_text = []
    index_map = []

    current_index = 0
    # Iterate over each character in the input text
    for original_index, char in enumerate(text):
        if char.isdigit():
            # If the character is a digit, convert it to words
            words = num2words(char)
            for word in words:
                for word_char in word:
                    # Append each character of the word to intermediate_text
                    intermediate_text.append(word_char.lower())
                    # Map the original index to the current index in intermediate_text
                    index_map.append((original_index, current_index))
                    current_index += 1
        else:
            word = unidecode(char)
            for word_char in word:
                # Append each character of the word to intermediate_text
                intermediate_text.append(word_char.lower())
                # Map the original index to the current index in intermediate_text
                index_map.append((original_index, current_index))
                current_index += 1

    # Join intermediate_text to form a single string
    intermediate_text = "".join(intermediate_text)



    # Initialize processed_text and final_index_map
    processed_text = []
    final_index_map = []
    new_index = 0

    # Iterate over each character in intermediate_text
    for original_index, char in enumerate(intermediate_text):
        if char.isalpha():
            # Keep only alphabetic characters, convert to lowercase
            processed_text.append(char.lower())
            # Map the original index to the new index in processed_text
            final_index_map.append((index_map[original_index][0], new_index))
            new_index += 1
        elif char == " ":
            # Replace spaces with '|'
            processed_text.append("|")
            # Map the original index to the new index in processed_text
            final_index_map.append((index_map[original_index][0], new_index))
            new_index += 1

    # Join processed_text to form the final string
    processed_text = "".join(processed_text)

    # # Print the mapping of original indices to new indices
    # for original_idx, new_idx in final_index_map:
    #     orig_char = repr(text[original_idx])
    #     proc_char = repr(processed_text[new_idx])
        # print(f"({original_idx:>12}, {new_idx:>8}), {orig_char:>13}, {proc_char:>14}")
    
    index_map_original_to_processed = {k:v for k,v in final_index_map}
    index_map_processed_to_original = {v:k for k,v in final_index_map}

    # Return the processed text and the final index mapping
    return processed_text, index_map_original_to_processed, index_map_processed_to_original
