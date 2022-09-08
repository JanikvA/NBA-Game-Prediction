# FROM cimg/python:3.10.6
# USER circleci
# WORKDIR /home/circleci/app
# COPY --chown=circleci . .

# FROM python:3.10.7-alpine3.16
# RUN apk add --no-cache curl bash musl-dev gcc build-base linux-headers gfortran

FROM ubuntu:latest
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get -y upgrade && apt-get -y install curl python3 python3-pip bash
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.0 python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry install
CMD /bin/bash
