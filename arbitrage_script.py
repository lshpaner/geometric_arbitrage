from marketspacedynamics import MarketSpaceDynamics
import matplotlib.pyplot as plt

plt.ion()  # enables interactive mode

############### Define the tickers, date range, and labels #####################
labels = {
    "CL=F": "NYMEX WTI Crude Oil",
    "BZ=F": "ICE Brent Crude Oil",
}

analysis = MarketSpaceDynamics(
    ["CL=F", "BZ=F"],
    "2022-01-01",
    "2024-01-01",
    forecast_steps=60,
    labels=labels,
)

######################## Plot the price-volume curves ##########################

analysis.plot_price_volume_curves(save_path="images/price_volume_curve.png")

########################### Plot the closing prices ############################

analysis.plot_closing_prices(save_path="images/closing_prices.png")

################################################################################
########################## Forecasting and Plotting ############################
################################################################################

forecast_dfs = analysis.plot_forecasts(
    steps=analysis.forecast_steps,
    save_path="images/price_forecast.png",
)

############### Extract Forecast DataFrames and Save to CSV ####################

## inspect the forecast dataframes

print(f"\nForecast BZ=F:\n{forecast_dfs['BZ=F'].head()}")  # inspect BZ=F forecast data
print(f"\nForecast CL=F:\n{forecast_dfs['CL=F'].head()}")  # inspect CL=F forecast data

forecast_bz_f = forecast_dfs["BZ=F"]
forecast_cl_f = forecast_dfs["CL=F"]

forecast_bz_f.to_csv("data/forecast_bz_f.csv")
forecast_cl_f.to_csv("data/forecast_cl_f.csv")

input("Press ENTER to quit...")
