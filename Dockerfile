FROM ubuntu

RUN apt-get update && \
	apt-get install curl python3 python3-distutils -y 
# pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Prereqs for multigrid
RUN pip install numpy gym matplotlib

# prereqs for ataripy which ray will install
RUN apt-get install cmake libz-dev -y && \
	apt install build-essential -y && \
	apt-get install manpages-dev -y

# prereqs for opencv which ray[rllib] needs to run an example
RUN apt-get install libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN pip install 'ray[rllib]' 

# Torch
RUN pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html


