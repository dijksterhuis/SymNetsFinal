FROM python:3.3-alpine
MAINTAINER dijksterhuis
RUN apk update
RUN apk upgrade
RUN apk add gcc g++ make libffi-dev openssl-dev
RUN pip install --upgrade pip
RUN pip install pycrypto
