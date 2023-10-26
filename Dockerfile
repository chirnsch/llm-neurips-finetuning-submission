FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
WORKDIR /code
COPY ./inference/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt update
RUN apt -y install wget
RUN wget https://llm-neurips-finetuning.s3.eu-central-1.amazonaws.com/mistral_finetuned.zip -O tmp.zip
RUN apt -y install unzip
RUN unzip tmp.zip
COPY ./inference /code/inference
CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "80"]
