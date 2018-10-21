FROM arm64v8/python:3.7-alpine

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./

ENV CONFIG_FILE "config.toml" 
ENTRYPOINT ["python"]
CMD ["API Weather.py"]