import asr.subtitles as sub

def write(subs):
    with open("../ngram-lm/data/wikitext-2/wiki.train.tokens", "w") as train,\
            open("../ngram-lm/data/wikitext-2/wiki.valid.tokens", "w") as valid,\
            open("../ngram-lm/data/wikitext-2/wiki.test.tokens", "w") as test,\
            open("../lexicon.txt", "w") as lexicon:   
        
        for line in subs.sentences:
            train.write(line)
            train.write("\n")

            valid.write(line)
            valid.write("\n")

            test.write(line)
            test.write("\n")

        for word in subs.unique_words:
            lexicon.write(word)
            for letter in word:
                lexicon.write(" " + letter)
            lexicon.write(" |")
            lexicon.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs='?', const=1, default=0, type=int)
    parser.add_argument("--end", nargs='?', const=1, type=int)
    args = parser.parse_args()
    
    subs = sub.Subtitles("data/spider_man_source.srt", args.start, args.end)
    write(subs, args.outdir)