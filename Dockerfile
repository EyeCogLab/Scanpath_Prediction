FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /app

ENV HTTP_PORT=4000
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

RUN apt-get update && apt-get -y install \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY ./requirements.txt ./requirements.txt

RUN python -m pip install --no-cache -U pip setuptools && \
    python -m pip install --no-cache -r requirements.txt && \
    python -m pip install --no-cache gunicorn orjson flask && \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" FORCE_CUDA=1 python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY ./hparams ./hparams
COPY ./irl_dcb ./irl_dcb
COPY ./trained_models ./trained_models
COPY ./model_server.py ./model_server.py
COPY ./all_task_ids.npy ./all_task_ids.npy

RUN python -m pip install --no-cache -U pip setuptools \
    && python -m pip install --no-cache -r requirements.txt \
    && python -m pip install gunicorn orjson flask

RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" FORCE_CUDA=1 python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

EXPOSE $HTTP_PORT

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "--pythonpath", ".", "--access-logfile", "-", "model_server:app"]