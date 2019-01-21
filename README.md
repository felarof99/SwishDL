
# SwishDL

## Description
This repo contains scripts and instructions to create your own Cloud Layer for efficiently running and scaling your DL training jobs.

Recommended setup:
1. Use sharded versions of CIFAR10, and ImageNet that are already available at below locations.   
    * _CIFAR10_: http://storage.googleapis.com/lpr-demo     
    * _ImageNet_: http://storage.googleapis.com/lpr-imagenet   
1. Create kubernetes cluster to meet your needs - GPU type (K80, P100, V100), Number of GPUs per node (1, 2, 4, 8).
1. Create a docker image of your trainer.
1. Use deploy script and launch your trainer (explained later).


## Overview of code structure:
- **dltrainer**
Contains Dockerfile and kubernetes deoployment file for PyTorch trainer.

- **cache-server**
Contains Dockerfile and kubernetes deployment file for NGINX cache-server.

- **ku** script:
Contains commands to setup kubernetes cluster on Google Cloud

- **kuber-cluster-config.sh** script:
Contains parameters that can be configured to customize the kubernetes setup

## Initial setup
Before proceeding, please ensure you have following packages installed locally; follow instructions available online.
1. Docker
1. Google Cloud SDK: After installing Google Cloud SDK, run `gcloud init`


## Kubernetes cluster setup:
1. Set parameters in `kube-cluster-config.sh`
2. Call `./ku init`. Please don't kill the execution inbetween. The command does the following:
    - Creates a kubernetes clusters
    - Creates a GPU node-pool with each node containing requested number of GPUs
    - Install cache-server and deploys it into the cluster

## Deploying PyTorch trainer on Kubernetes cluster:
Change to dltrainer folder.
1. Three important files are
    - train.py: PyTorch trainer
    - model.py: Neural Network model
    - dataset.py: Dataset loader. This file also serves as an example to fetch data using dlinputs library.
2. Build the docker image   
`docker build -t name_of_your_docker_image .`
3. Tag docker image   
`docker tag name_of_your_docker_image gcr.io/$GCLOUD_PROJECT_NAME/name_of_your_docker_image:v1` 
4. Upload docker image to a cloud repository.     
`gcloud docker -- push gcr.io/$GCLOUD_PROJECT_NAME/name_of_your_docker_image:v1`
5. Configure location of your image in trainer-deploy.yml
```
spec:
    template:
        spec:
          containers:
          - name: imagenet-training
            image: gcr.io/$GCLOUD_PROJECT_NAME/name_of_your_docker_image:v1
```

6. Configure other parameters required for your training job in `trainer-deploy.yml` - type of GPUs, number of GPUs.
``` yaml   
apiVersion: batch/v1
kind: Job
metadata:
  name: trainer-job
spec:
  template:
    spec:
      containers:
      - name: imagenet-training
        image: gcr.io/$GCLOUD_PROJECT_NAME/name_of_your_docker_image:v1 # TODO Put location of your image on cloud repository
        command: ["python"]
        args:
        - "train.py"
        - "--devices"
        - "1" # TODO Set the number of GPUs required by your Job        
        resources:          
          limits:
            nvidia.com/gpu: 1 # TODO Set this number to same as the number of GPUs required by your Job
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-k80 
      restartPolicy: Never
  backoffLimit: 4
```

7. Deploy your job   
`kubectl create -f trainer-job.yml`

## Profiler Dashboard
The base docker image also comes with a built-in profiler, which tracks GPU, CPU and Network Utilization every 30seconds. 

Below is a sample dashboard from training a network across 4 GPUs.

![Sample Dashboard](https://www.evernote.com/shard/s405/sh/c42efe14-ef62-481f-b196-64c3edde8cac/7620b9242b1d6462441f42b42c257f29/res/8ce9b60f-3372-4c3b-bbf6-99fb46f5005b.jpg)
