# Mask RCNN as a service
Mask Service for Image segmentation wrapped into [fastapi](https://fastapi.tiangolo.com/).  
The model is taken from this [github project](https://github.com/matterport/Mask_RCNN).  
Check it out to see what the model can do.

## Usage

Installing the dependencies for the RCNN on your system will be painful.  
This repo only support the execution using a docker container.  

The container image can be build using the provided [docker file](Dockerfile).  
```bash
docker build -t expingo-mask-service .
```

The build takes a while. Once finished, run
```
docker run -p 8000:8000 expingo-mask-service
```
You can inspect and play with the API in swagger
```
http://localhost:8000/docs
```

## Developer Guide
To create a usable development setting I recommend using the docker container and mounting the 
project into the container. 
```
docker run --rm -it -v /path/to/project/on/your/host:/dev -p 8000:8000 expingo-mask-service bash
```
Using swagger as describe above give an easy way to tease the api with some requests.