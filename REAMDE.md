sudo docker build -t image-name .
docker run --runtime=nvidia -it --mount type=bind,source="$(pwd)",target=/app image-name

docker ps -a
docker commit <container_id> image-name
