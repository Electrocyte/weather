FROM resin/raspberrypi3-python:3.6

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./

ENV CONFIG_FILE "config.toml" 
ENTRYPOINT ["python"]
CMD ["API Weather.py"]