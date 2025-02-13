# Build `inkpath` using `docker`

This docker image builds `inkpath` for Linux.
It is based on the same Ubuntu version as the GitHub Actions runner `ubuntu-latest`.

```sh
# Change docker build context and clean local build files
cd ..
# Build docker image
docker build -f ./docker/Dockerfile -t my_docker_image .
# Create temporary container and copy files to local machine
docker create --name temp_container my_docker_image
docker cp temp_container:/inkpath/build/ImageTranscription ./docker/ImageTranscription
docker cp temp_container:/inkpath/installed_packages.txt ./docker/installed_packages.txt
docker rm temp_container
# Remove docker image
docker rmi my_docker_image
```
