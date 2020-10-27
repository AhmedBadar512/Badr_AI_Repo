FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

RUN pip install opencv-python==4.1.2.30 && \
    pip install tensorflow_datasets && \
    pip install tensorflow_addons
RUN apt install -y libgtk2.0-dev libgl1-mesa-glx