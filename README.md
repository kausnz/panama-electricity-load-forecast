# Panama (Short-term) Electricity Load Forecast

## Problem Statement

The national electricity grid operator<sup>1</sup> in Panama requires a machine learning model to forecast the hourly
electricity demand<sup>2</sup> for a period of 7 days. This goes into a pre electricity dispatch<sup>3</sup> report released to the
electricity industry.
A dataset is available with hourly records. Its composition is as follows:

1. Historical electricity load, available on daily post-dispatch reports, from the grid
   operator ([CND](https://www.cnd.com.pa/), Panama).
2. Historical weekly forecasts available on weekly pre-dispatch reports, both from CND.
3. Calendar information related to school periods, from Panama's Ministry of Education.
4. Calendar information related to holidays, from "When on Earth?" website.
5. Weather variables, such as temperature, relative humidity, precipitation, and wind speed, for three main cities in
   Panama, from Earthdata.

Using this dataset, develop an ML model to forecast the electricity demand for 1-hour time window when a feature matrix
for the same time window is provided.

<sup>1</sup> The operating body who ensures the supplu-demand equilibrium in the national electricity grid.

<sup>2</sup> The amount of electricity 'demanded' by the consumers for a specific time window. In this dataset the time window is 1 hour. Also known as the *Load*. These two terms are use interchangeably throughout this document and the jupyter
notebook.

<sup>3</sup> Dispatch is the action of supplying electricity to the grid to meet a demand. This is done by electricity
generators. An instruction given by the grid operator to a generator, with specific instruction on how much to generate
and when, is known as a dispatch instruction.

### Dataset

Aguilar Madrid, Ernesto (2021), “Short-term electricity load forecasting (Panama case study)”, Mendeley Data, V1, doi:
10.17632/byx7sztj59.1
(https://data.mendeley.com/datasets/byx7sztj59/1)

## Solution

The solution consists of the following components:

* `notebook.ipynb` - A jupyter notebook that consists of an EDA of the dataset, model training, tuning and selection.
* `predict.py` - A REST service api for the model so the clients can get predictions calculated for a given set of
  features, over http. This service is hosted in Google Cloud Run and can be access by https://load-forecast-regressor-cloud-run-service-mqpvakdd5a-ue.a.run.app/#/Service%20APIs/load_forecast_regressor__forecast. Here's a sample request if you would like to test a forecast manually via the swagger ui.
    ```json
    {
      "t2m_toc": 25.6113220214844,
      "qv2m_toc": 0.01747758,
      "tql_toc": 0.043762207,
      "w2m_toc": 15.885400482402956,
      "t2m_san": 23.8613220214844,
      "qv2m_san": 0.016439982,
      "tql_san": 0.03894043,
      "w2m_san": 6.2321456709303815,
      "t2m_dav": 22.9472595214844,
      "qv2m_dav": 0.01531083,
      "tql_dav": 0.062301636,
      "w2m_dav": 3.6011136954933645,
      "holiday_id": 0.0,
      "holiday": 0.0,
      "school": 0.0,
      "dt_year": 2019.0,
      "dt_month": 1.0,
      "dt_day": 8.0,
      "dt_hour": 20.0
    }
    ```
* `train.py` - A script to train the best performing model and export as a bentoml model.
* `shared_func.py` - A shared module to have the xgboost training logic, so it can be called from both `notebook.ipynb`
  and `train.py`.
* `dashboard/app.py` - https://lf-dashboard-mqpvakdd5a-uc.a.run.app is an experimental but working dashboard I've developed to plot the forecast and the actual demand on the same
  graph.
  This dashboard sends a request to the model hosted in GCR every 2 seconds and plot the *forecast* value along with the
  corresponding *actual*. This helps to visualise how the forecast tracks compared to the actual demand.

Other files:

* `bentofile.yaml` - A bentoml descriptor to build, package and dockerize the application to a deployable unit.
* `deployment_config.yaml` - A configuration descriptor for `bentoctl` to deploy the service (predict.py, encapsulated
  in a bento package) to Google Cloud Run (GCR).
* `main.tf` and `terraform.tfstate` - Generated by bentoctl for the infrastructure to provision in GCR.
* `NOTES.md` - Some notes for self about the dataset.
* `Makefile` - A convenience script to organise frequently executed commands.

## System requirements to run locally

This notebook was developed on macOS platform. Pipenv is used to isolate the python version and dependencies, and they
can be found in the Pipfile and Pipfile.lock. You can initialise the python environment by
running `pipenv install --system --deploy`.

## Makefile

All the shell commands are collated into a Makefile for convenience. See the Makefile in this directory.

## Having trouble?

If you run into any issues when viewing the notebook or running the project locally, please reach out to me via
MLZoomcamp Slack (user @Kaush S.).