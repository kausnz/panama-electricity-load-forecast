main: nb2script

nb2script:
	echo "Converting the notebook to a python script"
	jupyter nbconvert --to script notebook.ipynb

serve:
	bentoml serve predict.py:load_forecast_svc

build-bento:
	bentoml build

build-docker:
	bentoml containerize load_forecast_regressor:latest