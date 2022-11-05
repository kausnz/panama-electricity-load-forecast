main: default

default:
	echo "choose a task"

## Model

nb2script:
	echo "Converting the notebook to a python script"
	jupyter nbconvert --to script notebook.ipynb

serve:
	bentoml serve predict.py:load_forecast_svc

build-bento:
	bentoml build

docker-build:
	bentoctl build -b load_forecast_regressor:latest -f deployment_config.yaml

deploy:
	terraform apply -var-file=bentoctl.tfvars -auto-approve

destroy:
	bentoctl destroy -f deployment_config.yaml

## Dashboard
dash-gcr: gcr-tag gcr-push

dash-build:
	docker build -t lf-dashboard ./dashboard/.

dash-run:
	docker run -it --rm -p 8050:8050 lf-dashboard

gcr-tag:
	docker tag lf-dashboard:latest gcr.io/load-forecast-regressor/lf-dashboard:latest

gcr-push:
	docker push gcr.io/load-forecast-regressor/lf-dashboard:latest