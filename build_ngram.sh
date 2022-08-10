if [ $1 -eq 0 ] && [ $2 -eq 0 ]
  then
    python3 ./asr/tokeniser.py --start=$1 --end=$2
else
    python3 ./asr/tokeniser.py
fi

cd ./ngram-lm

python3 ./main.py --order 3 --interpolate --save-arpa --name spiderman
python3 ./main.py --order 6 --interpolate --save-arpa --name spiderman