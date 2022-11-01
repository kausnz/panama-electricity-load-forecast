main: nb2script

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

