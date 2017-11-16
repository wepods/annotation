FROM ubuntu:16.04
LABEL maintainer fkariminejadasl@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        python \
        python-numpy \
        python-matplotlib \
        python-tk \
        python-yaml \
        python-opencv \
        libcanberra-gtk-module \
	      && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /home/annotate

# forward/backward to history
ADD inputrc /tmp
RUN cat /tmp/inputrc > ~/.inputrc
RUN rm /tmp/inputrc


# history sync from other terminal tabs. run histsync_import
ADD bashrc_append /tmp
RUN cat /tmp/bashrc_append >> ~/.bashrc
RUN rm /tmp/bashrc_append
