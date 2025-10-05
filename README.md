# Geometric Arbitrage: Market Space and Profit Distribution

This repository accompanies the research paper:

**“The Relationship Between Market Space and Profit Distribution in Arbitrage Situations: A Geometric Approach”**  
*by Leonid Shpaner*

## Overview

This project provides a practical implementation of the models and simulations discussed in the paper. It leverages historical crude oil futures data to explore the geometric structure of arbitrage opportunities through price-volume dynamics and forecasting.

The key objectives include:

- Modeling volume-sensitive price-response curves  
- Forecasting future price trends using ARIMA  
- Visualizing arbitrage regions based on geometric interpretations  
- Exporting data and figures for reproducibility
---

## Directory Structure

```text
geometric_arbitrage/
├── data/ # CSV forecast outputs (gitignored)
│ ├── forecast_bz_f.csv
│ └── forecast_cl_f.csv
│
├── images/ # Plot outputs (gitignored)
│ ├── closing_prices.png
│ ├── price_forecast.png
│ └── price_volume_curve.png
│
├── notebooks/
│ └── arbitrage_script.ipynb # Jupyter version of the workflow
│
├── arbitrage_script.py # Main runnable pipeline script
├── marketspacedynamics.py # Core logic class (MarketSpaceDynamics)
├── arbitrage_anonymized_supplementary.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Main Components

- **`marketspacedynamics.py`**  
  Core implementation of the `MarketSpaceDynamics` class, which performs:
  - Historical data fetching
  - ARIMA-based forecasting
  - Price-volume curve fitting
  - Plot generation and export

- **`arbitrage_script.py`**  
  Script entry point to run the full pipeline end-to-end.

- **`notebooks/arbitrage_script.ipynb`**  
  Interactive version of the pipeline for exploration in Jupyter.
---

## Visualizations

### Plot the price-volume curves

![](https://github.com/lshpaner/geometric_arbitrage/blob/main/images/price_volume_curve.png)

### Plot the closing prices

![](https://github.com/lshpaner/geometric_arbitrage/blob/main/images/closing_prices.png)

### Plot the forecasts

![](https://github.com/lshpaner/geometric_arbitrage/blob/main/images/price_forecast.png)

---

## Getting Started

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python arbitrage_script.py
```

3. Or use the Jupyter notebook:

```bash
jupyter notebook notebooks/arbitrage_script.ipynb
```

## Python Requirements

This project was developed and tested on **Python 3.11**.

All core dependencies are listed in the included `requirements.txt` file:

```text
matplotlib==3.10.6
numpy==1.26.4
pandas==2.3.3
pmdarima==2.0.4
scikit-learn==1.3.2
scipy==1.10.1
yfinance==0.2.66
```

Install them using:


```bash
pip install -r requirements.txt
```

---

## Citation

If you use or reference this work, please cite the associated paper.  
All material is provided for academic and research use.

---

## License

[MIT License](https://github.com/lshpaner/geometric_arbitrage/blob/main/LICENSE.md)

---

*Maintained by Leonid Shpaner.*
