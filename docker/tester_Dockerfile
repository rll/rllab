FROM neocxi/rllab_exp_gpu_tf:py3

RUN bash -c 'source activate rllab3 && conda install -y nomkl && conda uninstall -y scipy && conda install -y scipy'

ADD . /root/code/rllab
WORKDIR /root/code/rllab
