# FROM cimg/python:3.10.6
# USER circleci
# WORKDIR /home/circleci/app
# COPY --chown=circleci . .

FROM python:3.10.7-alpine3.16
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y curl bash
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install
CMD /bin/bash
