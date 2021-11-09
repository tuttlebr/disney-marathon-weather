import plotly.offline as py

py.init_notebook_mode()

from logging import basicConfig, info

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly

pd.options.mode.chained_assignment = None

basicConfig(
    format="%(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def forecast():
    DATE_COL = "DATE"
    PREDICTIONS = ["TMIN", "TMAX"]
    USECOLS = [DATE_COL] + PREDICTIONS
    NOAA_CSV = "/app/src/main/python/2776859.csv"
    MARATHON_DATES = [
        "2022-01-09",
        "2021-01-10",
        "2020-01-12",
        "2019-01-13",
        "2018-01-07",
        "2017-01-08",
        "2016-01-10",
        "2015-01-11",
        "2014-01-12",
        "2013-01-13",
        "2012-01-08",
        "2011-01-09",
        "2010-01-10",
        "2009-01-11",
        "2008-01-13",
        "2007-01-07",
        "2006-01-08",
        "2005-01-09",
        "2004-01-11",
        "2003-01-12",
        "2002-01-06",
        "2001-01-07",
        "2000-01-09",
        "1999-01-10",
        "1998-01-11",
        "1997-01-05",
        "1996-01-07",
        "1995-01-08",
        "1994-01-16",
    ]

    df = pd.read_csv(
        NOAA_CSV, usecols=USECOLS, infer_datetime_format=True
    ).interpolate()
    df.rename(columns={DATE_COL: "ds"}, inplace=True)
    df.head()

    manifest = {}
    for p in PREDICTIONS:
        info(f"Running forecast model training for {p}")
        m = Prophet(mcmc_samples=0)
        df_tmp = df[["ds", p]]
        df_tmp.rename(columns={p: "y"}, inplace=True)
        m.fit(df_tmp)
        future = m.make_future_dataframe(periods=70)
        forecast = m.predict(future)
        forecast = forecast.join(df_tmp, rsuffix="DROP").filter(regex="^(?!.*DROP)")
        # plot = plot_plotly(m, forecast, figsize=(1600, 900))
        forecast["ds"] = pd.to_datetime(forecast.ds)
        m.plot(forecast, figsize=(16, 9)).savefig(f"{p}.png")
        manifest[p] = {"forecast": forecast, "model": m}

    manifest["marathon_dates"] = MARATHON_DATES
    return manifest


if __name__ == "__main__":
    manifest = forecast()
    info("            ***** Anticipated Max Temps *****            ")
    info(
        manifest["TMAX"]["forecast"][
            manifest["TMAX"]["forecast"].ds.isin(manifest["marathon_dates"])
        ][["ds", "y", "yhat_lower", "yhat", "yhat_upper"]]
    )
    info("            ***** Anticipated Min Temps *****            ")
    info(
        manifest["TMIN"]["forecast"][
            manifest["TMIN"]["forecast"].ds.isin(manifest["marathon_dates"])
        ][["ds", "y", "yhat_lower", "yhat", "yhat_upper"]]
    )
