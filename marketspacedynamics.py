import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pmdarima import auto_arima


class MarketSpaceDynamics:
    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        forecast_steps=365,
        labels=None,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_steps = forecast_steps
        self.labels = (
            labels if labels is not None else {ticker: ticker for ticker in tickers}
        )
        self.data = {}
        self.fetch_data()
        self.preprocess_data()

    def fetch_data(self):
        for ticker in self.tickers:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
            )
            data = data.asfreq("B")  # Set frequency to business day
            data = data.ffill()  # Forward fill missing values
            self.data[ticker] = data

    def preprocess_data(self):
        for ticker in self.tickers:
            self.data[ticker]["Price"] = self.data[ticker]["Close"]
            self.data[ticker]["Volume"] = self.data[ticker]["Volume"]

    def price_volume_function(self, data, x):
        return np.interp(x, data["Volume"], data["Price"])

    def calculate_area(self, func, lower_limit, upper_limit):
        area, _ = quad(func, lower_limit, upper_limit)
        return area

    ########################### Plotting and forecasting #######################

    def save_plot_if_needed(self, save_path):
        """
        Save current matplotlib plot to the specified path if provided.
        """
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def plot_price_volume_curve(self, func, lower_limit, upper_limit, label):
        x_values = np.linspace(lower_limit, upper_limit, 100)
        y_values = func(x_values)
        plt.plot(x_values, y_values, label=label)
        plt.fill_between(x_values, y_values, alpha=0.2)

    def plot_price_volume_curves(self, save_path=None):
        # Get min and max volumes safely
        volume_min = min(
            float(self.data[ticker].filter(like="Volume").iloc[:, 0].min())
            for ticker in self.tickers
        )
        volume_max = max(
            float(self.data[ticker].filter(like="Volume").iloc[:, 0].max())
            for ticker in self.tickers
        )
        volume_range = np.linspace(volume_min, volume_max, 100)

        plt.figure(figsize=(10, 6))
        for ticker in self.tickers:
            # Grab first matching 'Volume' and 'Price' columns for this ticker
            vol = self.data[ticker].filter(like="Volume").iloc[:, 0]
            price = self.data[ticker].filter(like="Price").iloc[:, 0]
            # Build a small DataFrame to pass to your existing function
            df = pd.DataFrame({"Volume": vol, "Price": price}).dropna()

            self.plot_price_volume_curve(
                lambda x: self.price_volume_function(df, x),
                volume_min,
                volume_max,
                self.labels[ticker],
            )

        plt.xlabel("Volume")
        plt.ylabel("Price")
        plt.title("Price-Volume Curves")
        plt.legend()
        self.save_plot_if_needed(save_path)
        plt.show()

    def plot_closing_prices(self, save_path=None):
        plt.figure(figsize=(10, 6))
        for ticker in self.tickers:
            plt.plot(
                self.data[ticker].index,
                self.data[ticker]["Close"],
                label=f"{self.labels[ticker]} Historical",
            )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Closing Prices Over Time")
        plt.legend()
        self.save_plot_if_needed(save_path)
        plt.show()

    def forecast_prices(self, data, steps):
        model = auto_arima(
            data["Close"],
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            seasonal=True,
            m=12,
            D=1,
            stepwise=False,
            n_jobs=-1,
            suppress_warnings=True,
            trace=True,
        )

        # After fitting, print best model and its AIC
        print("Best model AIC:", model.aic())

        forecast = model.predict(n_periods=steps)
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B"
        )
        return pd.DataFrame({"Date": forecast_index, "Forecast": forecast})

    def plot_forecasts(self, steps, save_path=None):
        forecast_dfs = {}
        plt.figure(figsize=(10, 6))
        for ticker in self.tickers:
            forecast_df = self.forecast_prices(self.data[ticker], steps)
            forecast_dfs[ticker] = forecast_df
            plt.plot(
                self.data[ticker].index,
                self.data[ticker]["Close"],
                label=f"{self.labels[ticker]} Historical",
            )
            plt.plot(
                forecast_df["Date"],
                forecast_df["Forecast"],
                label=f"{self.labels[ticker]} Forecast",
                linestyle="--",
            )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Price Forecast")
        plt.legend()
        self.save_plot_if_needed(save_path)
        plt.show()
        return forecast_dfs

    def calculate_forecasted_areas(self, forecast_dfs):
        volume_range = np.linspace(
            min([self.data[ticker]["Volume"].min() for ticker in self.tickers]),
            max([self.data[ticker]["Volume"].max() for ticker in self.tickers]),
            100,
        )

        def price_volume_function_forecast(data, forecast_df, x):
            return np.interp(
                x,
                np.concatenate(
                    [data["Volume"], [data["Volume"].max()] * len(forecast_df)]
                ),
                np.concatenate([data["Price"], forecast_df["Forecast"]]),
            )

        forecast_areas = {}
        for ticker in self.tickers:
            forecast_areas[ticker] = self.calculate_area(
                lambda x: price_volume_function_forecast(
                    self.data[ticker], forecast_dfs[ticker], x
                ),
                volume_range.min(),
                volume_range.max(),
            )
        return forecast_areas

    def analyze_arbitrage_opportunities(self, areas):
        min_area = min(areas.values())
        max_area = max(areas.values())
        min_ticker = min(areas, key=areas.get)
        max_ticker = max(areas, key=areas.get)
        return {"buy_low": (min_area, min_ticker), "sell_high": (max_area, max_ticker)}

    def adjust_arbitrage_opportunities(self, opportunities, volume_range_max):
        opportunities["buy_low"] = (
            opportunities["buy_low"][0] / volume_range_max,
            opportunities["buy_low"][1],
        )
        opportunities["sell_high"] = (
            opportunities["sell_high"][0] / volume_range_max,
            opportunities["sell_high"][1],
        )
        return opportunities

    def interpret_results(self, opportunities, forecast=False):
        summary = "Profit Distribution Patterns: Analyzed profit distribution "
        summary += "patterns based on area differences.\n"
        summary += "Market Efficiency: Evaluated market efficiency based on "
        summary += "arbitrage opportunities.\n"

        if forecast:
            summary += "Forecasted "
        summary += "Arbitrage Opportunities:\n"
        if opportunities:
            summary += f"  Buy Low ({opportunities['buy_low'][1]}): "
            summary += f"{opportunities['buy_low'][0]}\n"
            summary += f"  Sell High ({opportunities['sell_high'][1]}): "
            summary += f"{opportunities['sell_high'][0]}\n"

        return summary

    def run_analysis(self):
        volume_range = np.linspace(
            min([self.data[ticker]["Volume"].min() for ticker in self.tickers]),
            max([self.data[ticker]["Volume"].max() for ticker in self.tickers]),
            100,
        )

        # Calculate current areas
        current_areas = {
            ticker: self.calculate_area(
                lambda x: self.price_volume_function(self.data[ticker], x),
                volume_range.min(),
                volume_range.max(),
            )
            for ticker in self.tickers
        }

        # Analyze current arbitrage opportunities
        current_opportunities = self.analyze_arbitrage_opportunities(current_areas)
        current_opportunities = self.adjust_arbitrage_opportunities(
            current_opportunities, volume_range.max()
        )
        current_summary = self.interpret_results(current_opportunities)

        # Forecast future prices
        forecast_dfs = self.plot_forecasts(self.forecast_steps)

        # Calculate forecasted areas
        forecast_areas = self.calculate_forecasted_areas(forecast_dfs)

        # Analyze forecasted arbitrage opportunities
        forecasted_opportunities = self.analyze_arbitrage_opportunities(forecast_areas)
        forecasted_opportunities = self.adjust_arbitrage_opportunities(
            forecasted_opportunities, volume_range.max()
        )
        forecasted_summary = self.interpret_results(
            forecasted_opportunities, forecast=True
        )

        # Print summaries
        print("Current Market Summary:")
        print(current_summary)

        print("\nForecasted Market Summary:")
        print(forecasted_summary)

        # Print forecasted dataframes
        for ticker, forecast_df in forecast_dfs.items():
            print(f"{ticker} Forecast for the next {self.forecast_steps} days:")
            print(forecast_df)

        return current_summary, forecasted_summary
