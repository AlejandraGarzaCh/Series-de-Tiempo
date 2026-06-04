# Documentación del Notebook: Random Walk


## Código 1

```python
import numpy as np
import pandas as pd
import plotly.express as px

# Parámetros
T = 500
sigma = 1
X0 = 0

# Ruido blanco normal
Z = np.random.normal(0, sigma, T)

# Construir Random Walk
X = np.zeros(T)
X[0] = X0

for t in range(1, T):
    X[t] = X[t-1] + Z[t]

# Crear DataFrame
df = pd.DataFrame({
    "Tiempo": np.arange(T),
    "X_t": X
})

# Gráfica
fig = px.line(
    df,
    x="Tiempo",
    y="X_t",
    title="Simulación de Random Walk",
    template="plotly_dark"   # <- aquí está lo bonito
)

# Línea horizontal en 0 (media teórica)
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="white",
    annotation_text="Media teórica = 0",
    annotation_position="top left"
)

# Ajustes estéticos
fig.update_layout(
    title_font_size=24,
    xaxis_title="Periodo (t)",
    yaxis_title="X_t",
    font=dict(size=14)
)

fig.show()


```


## Código 2

```python
# --- Instalar librerías (si no las tienes) ---
!pip -q install statsmodels

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# -----------------------------
# 1) Generar Random Walk
# Xt = X_{t-1} + Zt,  Zt ~ N(0, sigma^2)
# -----------------------------
np.random.seed(42)

T = 500
sigma = 1.0
X0 = 0.0

Z = np.random.normal(loc=0, scale=sigma, size=T)

X = np.zeros(T)
X[0] = X0
for t in range(1, T):
    X[t] = X[t-1] + Z[t]

# -----------------------------
# 2) Graficar la serie Xt
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(X, linewidth=2)
plt.title("Random Walk: $X_t = X_{t-1} + Z_t$")
plt.xlabel("t")
plt.ylabel("$X_t$")
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 3) Diferenciación: ΔXt = Xt - X_{t-1}
# -----------------------------
dX = np.diff(X)   # tamaño T-1

# -----------------------------
# 4) Graficar la serie diferenciada
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(dX, linewidth=2)
plt.title("Primera diferencia: $\\Delta X_t = X_t - X_{t-1}$")
plt.xlabel("t")
plt.ylabel("$\\Delta X_t$")
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 5) Tests de estacionariedad
#   - ADF: H0 = tiene raíz unitaria (NO estacionaria)
#   - KPSS: H0 = estacionaria (nivel)
# -----------------------------
def adf_test(series, name="Serie"):
    result = adfuller(series, autolag="AIC")
    print(f"\n--- ADF Test: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value      : {result[1]:.4f}")
    print("Regla rápida: p < 0.05 => Rechazas H0 (sin raíz unitaria) => estacionaria")
    return result

def kpss_test(series, name="Serie"):
    # regression='c' prueba estacionariedad alrededor de constante (nivel)
    stat, pvalue, lags, crit = kpss(series, regression='c', nlags="auto")
    print(f"\n--- KPSS Test: {name} ---")
    print(f"KPSS Statistic: {stat:.4f}")
    print(f"p-value       : {pvalue:.4f}")
    print("Regla rápida: p < 0.05 => Rechazas H0 (estacionaria) => NO estacionaria")
    return stat, pvalue, lags, crit

# Xt (esperas NO estacionaria)
adf_test(X, "X_t (Random Walk)")
kpss_test(X, "X_t (Random Walk)")

# ΔXt (esperas estacionaria; de hecho ΔXt ~ Zt)
adf_test(dX, "ΔX_t (Primera diferencia)")
kpss_test(dX, "ΔX_t (Primera diferencia)")

```

### 1. Ruido Blanco (White Noise)
Una serie de ruido blanco es una secuencia de variables aleatorias independientes e idénticamente distribuidas con una media de cero y varianza constante. No hay patrones ni autocorrelación.


## Código 4

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100

# Generar Ruido Blanco
white_noise = np.random.normal(loc=0, scale=1, size=T)

plt.figure(figsize=(12, 4))
plt.plot(white_noise, linewidth=2)
plt.title("Serie de Ruido Blanco")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Paseo Aleatorio (Random Walk)
Un paseo aleatorio es una serie no estacionaria donde el valor actual es igual al valor anterior más un término de ruido blanco. Se caracteriza por no volver a su media y su varianza aumenta con el tiempo. Ya tienes un ejemplo de esto en tu notebook.

### 3. Proceso Autorregresivo AR(1)
Un proceso AR(1) (Autoregressive de orden 1) es una serie de tiempo donde el valor actual depende linealmente de su propio valor anterior y un término de ruido blanco. Un $\phi$ entre -1 y 1 asegura estacionariedad.


## Código 7

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100
phi = 0.7  # Coeficiente AR(1)

# Generar ruido blanco
epsilon = np.random.normal(loc=0, scale=1, size=T)

# Construir serie AR(1)
ar_series = np.zeros(T)
ar_series[0] = epsilon[0] # Valor inicial
for t in range(1, T):
    ar_series[t] = phi * ar_series[t-1] + epsilon[t]

plt.figure(figsize=(12, 4))
plt.plot(ar_series, linewidth=2)
plt.title(f"Serie AR(1) con $\phi$ = {phi}")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. Proceso de Medias Móviles MA(1)
Un proceso MA(1) (Moving Average de orden 1) es una serie de tiempo donde el valor actual depende linealmente del término de ruido blanco actual y de un término de ruido blanco anterior.


## Código 9

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100
theta = 0.5  # Coeficiente MA(1)

# Generar ruido blanco
epsilon = np.random.normal(loc=0, scale=1, size=T)

# Construir serie MA(1)
ma_series = np.zeros(T)
for t in range(1, T):
    ma_series[t] = epsilon[t] + theta * epsilon[t-1]

plt.figure(figsize=(12, 4))
plt.plot(ma_series, linewidth=2)
plt.title(f"Serie MA(1) con $\theta$ = {theta}")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.show()
```

### 5. Serie de Tiempo Estacional
Una serie de tiempo estacional muestra patrones regulares y predecibles que se repiten en intervalos fijos (por ejemplo, diariamente, semanalmente, mensualmente, anualmente). Se compone de una tendencia, estacionalidad y un componente residual.


## Código 11

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 200 # Número de puntos
seasonal_period = 12 # Por ejemplo, 12 meses para estacionalidad anual

# Componente de tendencia
time = np.arange(T)
trend = 0.1 * time

# Componente estacional
seasonal = 10 * np.sin(2 * np.pi * time / seasonal_period)

# Componente de ruido blanco
noise = np.random.normal(loc=0, scale=2, size=T)

# Combinar para crear la serie estacional
seasonal_series = trend + seasonal + noise

plt.figure(figsize=(12, 4))
plt.plot(seasonal_series, linewidth=2)
plt.title("Serie de Tiempo Estacional")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.show()
```


## Código 12

```python
import numpy as np
import matplotlib.pyplot as plt

# -------- función MA(1) --------
def simulate_ma1(n, theta, sigma=1.0, seed=123):
    np.random.seed(seed)
    z = np.random.normal(0, sigma, n+1)
    x = z[1:] + theta * z[:-1]
    return x

n = 300

# =========================
# MODELO 1
# x_t = z_t + 2 z_{t-1}
# =========================
x1 = simulate_ma1(n, theta=2)

plt.figure(figsize=(12,4))
plt.plot(x1)
plt.title("Modelo 1:  x_t = z_t + 2 z_{t-1}")
plt.xlabel("t")
plt.ylabel("x_t")
plt.grid()
plt.show()


# =========================
# MODELO 2
# x_t = z_t + 1/2 z_{t-1}
# =========================
x2 = simulate_ma1(n, theta=0.5)

plt.figure(figsize=(12,4))
plt.plot(x2)
plt.title("Modelo 2:  x_t = z_t + 0.5 z_{t-1}")
plt.xlabel("t")
plt.ylabel("x_t")
plt.grid()
plt.show()


```
