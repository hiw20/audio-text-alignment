# Importing the necessary module 'sub' from the 'asr.subtitles' package
import subtitles as sub


# Defining a function named 'write' that takes a parameter 'subs'
def write(subs):
    # Opening files in write mode and assigning them to variables train, valid, test, and lexicon
    with open("./ngram-lm/data/wikitext-2/wiki.train.tokens", "w") as train, open(
        "./ngram-lm/data/wikitext-2/wiki.valid.tokens", "w"
    ) as valid, open("./ngram-lm/data/wikitext-2/wiki.test.tokens", "w") as test, open(
        "./lexicon.txt", "w"
    ) as lexicon:
        # Looping through each line in the 'sentences' attribute of 'subs'
        for line in subs.sentences:
            # Writing the line to the 'train' file
            train.write(line)
            # Writing a new line character to the 'train' file
            train.write("\n")

            # Writing the line to the 'valid' file
            valid.write(line)
            # Writing a new line character to the 'valid' file
            valid.write("\n")

            # Writing the line to the 'test' file
            test.write(line)
            # Writing a new line character to the 'test' file
            test.write("\n")

        # Looping through each word in the 'unique_words' attribute of 'subs'
        for word in subs.unique_words:
            # Writing the word to the 'lexicon' file
            lexicon.write(word)
            # Looping through each letter in the word
            for letter in word:
                # Writing a space character followed by the letter to the 'lexicon' file
                lexicon.write(" " + letter)
            # Writing a "|" character to the 'lexicon' file
            lexicon.write(" |")
            # Writing a new line character to the 'lexicon' file
            lexicon.write("\n")


# Checking if the current module is the main module
if __name__ == "__main__":
    # Importing the 'argparse' module
    import argparse

    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Adding an optional argument named 'start' to the parser
    parser.add_argument("--start", nargs="?", const=1, default=0, type=int)
    # Adding an optional argument named 'end' to the parser
    parser.add_argument("--end", nargs="?", const=1, type=int)
    # Adding an argument named 'filepath' to the parser
    parser.add_argument("--filepath", type=str)
    # Parsing the command-line arguments and storing them in 'args'
    args = parser.parse_args()

    # Creating a 'Subtitles' object named 'subs' by passing the file path and command-line arguments to the constructor
    subs = sub.Subtitles(args.filepath, args.start, args.end)
    # Calling the 'write' function and passing 'subs' as an argument
    write(subs)


# import asr.subtitles as sub

# def write(subs):
#     with open("../ngram-lm/data/wikitext-2/wiki.train.tokens", "w") as train,\
#             open("../ngram-lm/data/wikitext-2/wiki.valid.tokens", "w") as valid,\
#             open("../ngram-lm/data/wikitext-2/wiki.test.tokens", "w") as test,\
#             open("../lexicon.txt", "w") as lexicon:

#         for line in subs.sentences:
#             train.write(line)
#             train.write("\n")

#             valid.write(line)
#             valid.write("\n")

#             test.write(line)
#             test.write("\n")

#         for word in subs.unique_words:
#             lexicon.write(word)
#             for letter in word:
#                 lexicon.write(" " + letter)
#             lexicon.write(" |")
#             lexicon.write("\n")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--start", nargs='?', const=1, default=0, type=int)
#     parser.add_argument("--end", nargs='?', const=1, type=int)
#     args = parser.parse_args()

#     subs = sub.Subtitles("data/spider_man_source.srt", args.start, args.end)
#     write(subs, args.outdir)
