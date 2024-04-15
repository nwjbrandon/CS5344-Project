FROM ubuntu:18.04

# windows/linux
ARG ARCHITECTURE=x86_64
# macos
# ARG ARCHITECTURE=aarch64
ARG SPARK_VERSION=3.0.1
ARG HADOOP_VERSION_SHORT=3.2
ARG HADOOP_VERSION=3.2.0

WORKDIR /tmp

# Install java
RUN apt-get update 
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install vim wget -y

# Install spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION_SHORT}.tgz
RUN tar xvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION_SHORT}.tgz --directory /usr/local
ENV PATH="${PATH}:/usr/local/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION_SHORT}/bin"

# Install python3.6
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCHITECTURE}.sh \
&& mkdir /root/.conda \
&& bash Miniconda3-latest-Linux-${ARCHITECTURE}.sh -b \
&& rm -f Miniconda3-latest-Linux-${ARCHITECTURE}.sh

COPY environment.yml .
RUN conda env create -f environment.yml -y

WORKDIR /data
RUN echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate CS5344" >> ~/.bashrc
