FROM ubuntu:20.04


RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt install -y software-properties-common 
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install python3.9 -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 3
RUN apt install python3-pip -y

RUN mkdir /app

RUN apt update
RUN apt install git -y
RUN git clone https://github.com/daandouwe/ngram-lm.git /app

COPY ./requirements.txt /app
RUN pip install -r app/requirements.txt

COPY . /app

RUN cd /app && python3 /app/wav2vec.py --start=0 --end=100

RUN rm -rf /app
RUN mkdir /app