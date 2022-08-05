import subtitles as sub

def write(subs):
    with open("ngram-lm/data/wikitext-2/wiki.train.tokens", "w") as train,\
            open("ngram-lm/data/wikitext-2/wiki.valid.tokens", "w") as valid,\
            open("ngram-lm/data/wikitext-2/wiki.test.tokens", "w") as test,\
            open("lexicon.txt", "w") as lexicon:   
        
        for line in subs.sentences:
            train.write(line)
            train.write(" ")

            valid.write(line)
            valid.write(" ")

            test.write(line)
            test.write(" ")

        for word in subs.unique_words:
            lexicon.write(word)
            for letter in word:
                lexicon.write(" " + letter)
            lexicon.write(" |")
            lexicon.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    print(args.start, args.end)
    
    subs = sub.Subtitles("data/spider_man_source.srt", args.start, args.end)
    write(subs)