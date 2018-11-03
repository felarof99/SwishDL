FROM nvidia/cuda:9.0-base
MAINTAINER Tom Breuel <tmbdev@gmail.com>

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-get -y update && apt-get dist-upgrade -y

RUN  apt-get -y install sudo lsb-release build-essential curl software-properties-common \
    && echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" \
           >> /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update -y && apt-get dist-upgrade -y && apt-get -y install apt-utils

RUN apt-get -y install locales && locale-gen en_US.UTF-8 && dpkg-reconfigure locales \
    && apt-get -y install google-cloud-sdk

RUN apt-get -y install wget tightvncserver tmux rxvt \
    xterm mlterm imagemagick firefox blackbox imagemagick \
    vim-gtk gnome-terminal i3 chromium-browser git mercurial lynx daemon

RUN apt-get install -y nginx
RUN apt-get install -y nginx-extras
RUN apt-get install -y cadaver

RUN cd /tmp \
    && wget --quiet -nd https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
    && bash ./Miniconda2-latest-Linux-x86_64.sh -b -p /opt/conda \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && ln -s /opt/conda/bin/conda /usr/bin/conda \
    && rm -f Miniconda*.sh

RUN apt-get install -y redis-tools
RUN apt-get install -y bwm-ng

RUN conda install git
RUN conda install numpy
RUN conda install scipy
RUN conda install msgpack
RUN conda install simplejson
RUN conda install pyzmq
RUN conda install jupyter
RUN conda install scikit-image
RUN conda install scikit-learn
RUN conda install redis
RUN conda install Pillow
RUN conda install pytorch=0.4.0 torchvision cuda90 -c pytorch
RUN conda install msgpack
RUN conda install -c conda-forge google-cloud-storage
RUN conda install cython
RUN conda install -c auto editdistance

RUN conda install -c menpo opencv3
RUN conda install pip && ln -s /opt/conda/bin/pip /usr/bin/pip
RUN conda install setuptools
RUN conda install future
RUN conda install tensorflow
RUN pip install git+https://github.com/lanpa/tensorboard-pytorch
RUN pip install apscheduler
RUN pip install gpustat

ENV PATH /opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY dlinputs /tmp/dlinputs
RUN cd /tmp/dlinputs && python setup.py install && rm -rf /tmp/dlinputs

ENV USER user
ENV HOME /home/$USER
ENV GID 1000
ENV UID 1000

ENV TERM xterm
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir -p $HOME && groupadd -g $GID -r $USER && useradd --no-log-init -u $UID -r -g $USER $USER

RUN true \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> $HOME/.bashrc \
    && echo "conda activate base" >> $HOME/.bashrc \
    && chown -R $USER.$USER $HOME
RUN echo 'user ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $UID

RUN mkdir /home/$USER/large_scale_imagenet/
RUN mkdir /home/$USER/large_scale_imagenet/data/
RUN mkdir /home/$USER/large_scale_imagenet/testdata/

WORKDIR /home/$USER/large_scale_imagenet/

COPY ./*.py ./
# COPY ./testdata/*.tgz ./testdata/

# Setting cache-server as the proxy
ENV http_proxy http://cache-layer-service:3128