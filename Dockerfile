#Dockerfile
FROM python:3.6-buster
#FROM python:3.6-alpine

RUN apt-get update
RUN apt-get -y install libglib2.0
RUN apt-get clean

#RUN apk update
#RUN apk add glib-dev

COPY . /app
WORKDIR /app
COPY requirements.txt /app


RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 4000
CMD ["python", "app.py"]~
