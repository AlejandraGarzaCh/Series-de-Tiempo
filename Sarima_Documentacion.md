# Documentación del Notebook SARIMA


## Código 1

```python
# MODELO AIRLINE: SARIMA (0,1,1)(0,1,1)12

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# 1. Crear una serie mensual de ejemplo
# ---------------------------
np.random.seed(42)

n = 120  # 10 años de datos mensuales
fechas = pd.date_range(start='2015-01-01', periods=n, freq='MS')

# Serie con tendencia suave + estacionalidad + ruido
t = np.arange(n)
serie = 100 + 0.3*t + 8*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, n)

y = pd.Series(serie, index=fechas)

# ---------------------------
# 2. Ajustar modelo SARIMA (0,1,1)(0,1,1,12)
# ---------------------------
modelo = SARIMAX(
    y,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado = modelo.fit(disp=False)

# Resumen del modelo
print(resultado.summary())

# ---------------------------
# 3. Predicción dentro de muestra
# ---------------------------
pred = resultado.fittedvalues

# ---------------------------
# 4. Forecast a 24 meses
# ---------------------------
forecast_steps = 24
pronostico = resultado.get_forecast(steps=forecast_steps)
mean_forecast = pronostico.predicted_mean
conf_int = pronostico.conf_int()

# ---------------------------
# 5. Gráfica
# ---------------------------
plt.figure(figsize=(14,6))
plt.plot(y, label='Serie observada')
plt.plot(mean_forecast, label='Pronóstico 24 meses')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 alpha=0.3)

plt.title('Modelo Airline: SARIMA (0,1,1)(0,1,1)12')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()
```
