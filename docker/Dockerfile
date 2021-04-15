FROM tensorflow/tensorflow:latest-gpu

# RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install opencv-python==4.1.2.30 && \
    pip install tensorflow_datasets && \
    pip install tensorflow_addons && \
    pip install tensorflow_probability && \
    pip install psutil
RUN apt update --fix-missing
RUN apt install -y libgtk2.0-dev libgl1-mesa-glx
#RUN rm -rf /usr/bin/python && \
#    ln -s /usr/bin/python3.7 /usr/bin/python
#RUN mkdir /.horovod/
#RUN chmod -R 777 /.horovod/
WORKDIR /workspace/
#RUN git clone https://github.com/AhmedBadar512/Badr_AI_Repo.git -b collect-imgclsm-models /workspace
