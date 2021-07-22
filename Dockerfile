FROM python:slim
WORKDIR /app
COPY . .
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3-opencv && pip3 install -r requirements.txt
ENTRYPOINT ["python3","./main.py"]
