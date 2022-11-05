FROM python:3.9-slim-buster

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["test_input.json", "app.py", "./"]

EXPOSE 8050

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8050", "app:server" ]