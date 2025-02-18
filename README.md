# Air Quality Forecasting using Prophet

## Overview
This project aims to forecast air quality using the **Facebook Prophet** time series forecasting model. The dataset contains air quality measurements, including **PM2.5, PM10, CO, NO2, and other pollutants**, along with timestamps. The model is trained to predict future air quality levels based on historical data.

## Dataset
The dataset includes the following key columns:
- `ds` : Timestamp of the air quality measurement
- `y` : Air quality index (AQI) or specific pollutant concentration
- Additional features: Temperature, Humidity, Wind Speed, etc.

## Installation
Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy matplotlib prophet
```

## Data Preprocessing
Before training the model, the dataset is cleaned and preprocessed:
1. Convert the `ds` column to datetime format.
   ```python
   import pandas as pd
   data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
   ```
2. Handle missing values:
   ```python
   data = data.dropna(subset=['ds', 'y'])
   ```
3. Filter out unreasonable dates:
   ```python
   data = data[(data['ds'] >= '2000-01-01') & (data['ds'] <= '2030-01-01')]
   ```

## Model Training
The forecasting model is implemented using Facebook Prophet:
```python
from prophet import Prophet

model = Prophet()
model.fit(data)
```

## Forecasting Future Values
Once trained, the model can predict future air quality levels:
```python
future = model.make_future_dataframe(periods=30, freq='D')
forecast = model.predict(future)
```

## Visualization
Plot the forecasted values:
```python
import matplotlib.pyplot as plt
model.plot(forecast)
plt.show()
```

## Usage
This model can be used to:
- Predict air quality for the next `n` days.
- Identify trends and seasonal variations in air pollution levels.
- Assist in environmental policy-making and public health decisions.

## Issues & Debugging
- If you encounter an **OutOfBoundsDatetime** error, ensure `ds` values are within a reasonable date range.
- If `NaT` (Not a Time) values appear, drop or replace them before training the model.

## Contributors
- **Atharva Surve**

## License
This project is open-source and available under the MIT License.


