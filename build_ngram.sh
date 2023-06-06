# cd ..

if [ $3 -eq 0 ] && [ $4 -eq 0 ]
  then
    python3 ./asr/tokeniser.py --start=$3 --end=$4 --filepath=$1
else
    python3 ./asr/tokeniser.py --filepath=$1
fi

cd ./ngram-lm

# python3 ./main.py --order 3 --interpolate --save-arpa --name $2
python3 ./main.py --order 6 --interpolate --save-arpa --name $2

cd ..
