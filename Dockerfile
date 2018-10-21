FROM resin/raspberrypi3-python:3.6

WORKDIR /app

ARG PANDAS_VERSION=0.23.4
RUN pip install pandas==${PANDAS_VERSION}

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./

ENV CONFIG_FILE "config.toml"
ENTRYPOINT ["python"]
CMD ["API Weather.py"]
