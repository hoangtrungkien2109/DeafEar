FROM python:3.11

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /deafear

COPY pyproject.toml poetry.lock ./
COPY deafear ./deafear
RUN touch README.md

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "unit_test.py"]