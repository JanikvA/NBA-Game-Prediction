FROM cimg/python:3.10.6

RUN mkdir src/
COPY . src/

COPY pyproject.toml pyproject.toml
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install


CMD ["poetry shell"]
