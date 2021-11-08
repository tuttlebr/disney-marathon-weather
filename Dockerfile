FROM --platform=amd64 python:3.8

WORKDIR /app

COPY . .

RUN pip install pip install pystan==2.19.1.1 \
    && pip install --no-cache-dir -r \
    src/main/python/requirements.txt