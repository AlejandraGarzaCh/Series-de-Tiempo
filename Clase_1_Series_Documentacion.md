# Documentación del Notebook: Clase 1 - Series de Tiempo


## Código 1

```python
import pandas as pd

ventas = pd.read_csv("california_housing_train.csv")
ventas.head(51)
```


## Código 2

```python
import yfinance as yf
amzn = yf.download("AMZN", end="2026-01-01")
amzn
```


## Código 3

```python
precios_amzn = amzn["Close"]
precios_amzn
```


## Código 4

```python
import plotly.express as px

fig = px.line(precios_amzn, x=precios_amzn.index, y='AMZN', title='AMZN Close Price Over Time')
fig.show()
```
