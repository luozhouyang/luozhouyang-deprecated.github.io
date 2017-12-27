---
layout: post
title:  "使用Docker搭建Tensorflow开发环境"
date:   2017-12-04 21:15:05 +0000
image: /assets/images/startup.jpg

---
本文基于Ubuntu 16.04 LTS使用Docker来搭建Tensorflow的开发环境.  

Docker和Tensorflow就不用介绍了,项目均已开源在Github.  
* [Tensorflow](https://github.com/tensorflow/tensorflow)  
* [Docker CE](https://github.com/docker/docker-ce)  

## Docker的安装  
Docker的安装在官方网站有非常详细的文档,请参考:[Docker CE installation on Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)  
本文只是做一个稍微简单一点的概述.  

### Step 1 卸载旧版本  
使用如下命令:  

>> **sudo apt remove docker docker-engine docker.io**  


### Step 2 安装Docker CE  
依次使用如下命令:  

```bash  
sudo apt update  
sudo apt install \  
    apt-transport-https \  
    ca-certificates \  
    software-properties-common  
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -  
# 验证指纹  
sudo apt-key fingerprint 0EBFCD88  
# X86_64/amd64平台    
sudo add-apt-repository \  
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \  
   $(lsb_release -cs) \  
   stable"  
# 安装Docker CE  
sudo apt update  
sudo apt install docker-ce  
```   

至此,Docker CE已经通过apt源安装在机器上了.  

## Tensorflow的安装  
接下来可以安装Tensorflow了.  

本教程使用的是Tensorflow GPU版本,因此需要在Docker Hub拉取Tensorflow-gpu的镜像即可.  
如果安装CPU版本的Tensorflow,会更简单.  

### 安装NVIDIA显卡驱动和CUDA toolkit  
目前Tensorflow GPU版本支持需要CUDA 8.0和cuDNN 6.0, 直接从官网下载安装好即可.  
驱动直接安装最新的即可.  
CUDA 8.0的下载入口不明显,在NVIDIA下载页面的底部,有一个 **Legacy Releases** 链接,点进去既可看到各种历史版本的CUDA.  
cuDNN安装6.0版本即可.  
安装好之后要配置环境变量.  

### 安装NVIDIA-Docker  

```bash  
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge nvidia-docker
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd  
```  

### 拉取并运行Tensorflow-GPU镜像  

```bash  
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu  
```  

本地没有 tensorflow/tensorflow:latest-gpu镜像,Docker会自动从Docker Hub拉取.  
等它结束之后,本地已经有了最新版本的GPU版的Tensorflow.  
上述命令会进入一个新的容器,里面装好了tensorflow-gpu版本,可以直接使用了.  



