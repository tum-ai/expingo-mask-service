# Mask RCNN as a service
Mask Service for Image segmentation wrapped into [fastapi](https://fastapi.tiangolo.com/).  

## Usage

Installing the dependencies for the RCNN on your system will be painful.  
This repo only support the execution using a docker container.  

The container image can be build using the provided [docker file](Dockerfile).  
```bash
docker build -t expingo-mask-service .
```

If you build the docker container without the weight the first time, they will be downloaded 
once the container is executed but it takes some time.  
You can start the container with interactive mode
```
docker run --rm -it expingo-mask-service
```
and run the app script in the container
```
python3.7 main.py
```
Then copy the file out of the container (run this on your host system)
```bash
docker cp {container_id}:/app/mask_rcnn_coco.h5 .
```
You can find the container id with
```
docker ps
```
Then rebuild the container while placing the file in the project root directory.