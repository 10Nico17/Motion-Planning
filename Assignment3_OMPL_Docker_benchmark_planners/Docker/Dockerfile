FROM kavrakilab/omplapp

USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib pyyaml meshcat --upgrade

WORKDIR /home/ompl

USER ompl

CMD ["bash"]
