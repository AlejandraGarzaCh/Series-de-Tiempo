# Documentación del Notebook: Amazon Series de Tiempo


## Código 1

```python
import yfinance as yf
import pandas as pd
import numpy as np
```


## Código 2

```python
amzn = yf.download("AMZN",start="2010-01-01")
amzn
```


## Código 3

```python
amzn_precios = amzn["Close"]
amzn_precios
```


## Código 4

```python
import altair as alt

df = amzn_precios.reset_index()
df.columns = ["Date", "Close"]

chart = alt.Chart(df).mark_line().encode(
    x=alt.X("Date:T", title="Fecha"),
    y=alt.Y("Close:Q", title="Precio de cierre"),
    tooltip=["Date:T", "Close:Q"]
).properties(
    title="Serie de tiempo AMZN",
    width=900,
    height=400
).interactive()

chart
```


## Código 5

```python
log_returns = np.log(df['Close'].pct_change() + 1)
print("Log Returns:")
display(log_returns.head())
```
