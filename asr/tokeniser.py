from unidecode import unidecode
from num2words import num2words

def tokenize(text):
    """
    Tokenizes text by removing diacritics, replacing digits with words, and keeping only 
    alphabetic characters or spaces. Words are separated by "|" characters.
    """

    # Replace digits with words
    text = " ".join([num2words(s) if s.isdigit() else s for s in text.split()])

    # Remove diacritics (e.g., accents, umlauts)
    text = unidecode(text)

    # Keep only alphabetic characters or spaces, convert to lowercase
    text = [i.lower() if (i.isalpha() or i == " ") else "" for i in text]

    # Join characters and replace spaces with "|"
    text = "".join(text).replace(" ", "|")
    
    return text
