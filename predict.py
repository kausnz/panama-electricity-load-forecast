import bentoml

from bentoml.io import JSON
from pydantic import BaseModel


class FeaturePayload(BaseModel):
    t2m_toc: float
    qv2m_toc: float
    tql_toc: float
    w2m_toc: float
    t2m_san: float
    qv2m_san: float
    tql_san: float
    w2m_san: float
    t2m_dav: float
    qv2m_dav: float
    tql_dav: float
    w2m_dav: float
    holiday_id: int
    holiday: int
    school: int
    dt_year: int
    dt_month: int
    dt_day: int
    dt_hour: int


model_ref = bentoml.xgboost.get("load_forecast_model:latest")
model_runner = model_ref.to_runner()
dv = model_ref.custom_objects['dictVectorizer']

# this variable name is what we refer to when running `bentoml serve ...` to start the service
load_forecast_svc = bentoml.Service("load_forecast_regressor", runners=[model_runner])


@load_forecast_svc.api(input=JSON(pydantic_model=FeaturePayload), output=JSON())
async def forecast(payload: FeaturePayload):
    application_data = payload.dict()

    # note that we are vector-ising but not transforming to a dictionary afterwards.
    # the runner must be doing the to_dict(orient='records') internally. And also note
    # the feature_names are not passed in either.
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    return prediction[0]
