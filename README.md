WARNING - RUNNING THIS CODE WILL TAKE OVER 9 HOURS INCLUDING THE 
BUILD FOR THE DOCKER IMAGE.

To run this code you need to use docker.

1. Run the command below in the root directory of this project

docker build -t image-name .

2. Run the command below to run the image in an interactive shell. This is needed
to be able to run the code. Remove --runtime=nvidia if you do not want to run it with
a GPU.

docker run --runtime=nvidia -it --mount type=bind,source="$(pwd)",target=/app image-name

3. Once in the interactive shell change directory to the /app folder with

cd /app

4. Run the transcription-alignment program. The start and end parameters are used to specify
which subtitles to split the audio on. If you want to run on the complete audio then use 
start=0 and end=10000. Any large number will work for the end parameter, it just has to be larger
than the final subtitle index.

python3 ./generate_data.py --start=0 --end=10000

5. Running the above command will generate the predicted subtitles. This will take a long time,
almost 10 hours on a good GPU. This will also download all of the necessary ASR models. To make sure
the models don't need to be downloaded again you can run the two commands below to save the container
as a new image.

Open a new terminal but do not close the interactive docker shell that is open. The first command will 
list all the images and active containers. Find the container ID of the running interactive shell.

docker ps -a

6. Use the container ID to update the image from earlier. Now when you run the image it will have 
everything downloaded.

docker commit <container_id> image-name

7. To produce the figures run the command

python3 ./generate_metrics.py --start=0 --end=10000
