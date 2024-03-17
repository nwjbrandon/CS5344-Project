FROM ubuntu:18.04

# windows/linux
# ARG ARCHITECTURE=x86_64
# macos
ARG ARCHITECTURE=aarch64


WORKDIR /tmp

# Install java
RUN apt-get update 
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install vim wget -y

# Install scala
RUN wget https://downloads.lightbend.com/scala/2.12.4/scala-2.12.4.tgz 
RUN tar xvf scala-2.12.4.tgz --directory /usr/local
ENV PATH="${PATH}:/usr/local/scala-2.12.4/bin"

# Install maven
RUN wget https://archive.apache.org/dist/maven/maven-3/3.5.2/binaries/apache-maven-3.5.2-bin.tar.gz
RUN tar xvf apache-maven-3.5.2-bin.tar.gz --directory /usr/local
ENV PATH="${PATH}:/usr/local/apache-maven-3.5.2/bin"

# Install spark
RUN wget https://archive.apache.org/dist/spark/spark-2.2.1/spark-2.2.1-bin-hadoop2.7.tgz
RUN tar xvf spark-2.2.1-bin-hadoop2.7.tgz --directory /usr/local
ENV PATH="${PATH}:/usr/local/spark-2.2.1-bin-hadoop2.7/bin"

# Install python3.6
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCHITECTURE}.sh \
&& mkdir /root/.conda \
&& bash Miniconda3-latest-Linux-${ARCHITECTURE}.sh -b \
&& rm -f Miniconda3-latest-Linux-${ARCHITECTURE}.sh
RUN conda create -n CS5344 python=3.7 -y

WORKDIR /data
RUN echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate CS5344" >> ~/.bashrc
