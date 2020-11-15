FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install opencv-python==4.1.2.30 && \
    pip install tensorflow_datasets && \
    pip install tensorflow_addons && \
    pip install tensorflow
RUN apt update --fix-missing
RUN apt install -y libgtk2.0-dev libgl1-mesa-glx
RUN rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3.7 /usr/bin/python
WORKDIR /workspace/
RUN git clone https://github.com/AhmedBadar512/Badr_AI_Repo.git -b collect-imgclsm-models /workspace