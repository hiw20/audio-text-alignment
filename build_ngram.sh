python3 ./tokeniser.py --start=$1 --end=$2

cd ./ngram-lm

python3 ./main.py --order 6 --interpolate --save-arpa --name spiderman