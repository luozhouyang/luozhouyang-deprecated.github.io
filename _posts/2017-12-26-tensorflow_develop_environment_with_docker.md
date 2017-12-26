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

