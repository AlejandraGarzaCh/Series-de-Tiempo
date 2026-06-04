# Documentación del Notebook: Series de Tiempo


## Celda de Código 1

```python
import pandas as pd
import yfinance as yf

```


<!-- Markdown Cell 2 -->

### Visualización de la Serie Temporal Generada

Primero, vamos a recordar cómo se ve la serie `ar1_series` que hemos simulado. Esto nos dará una base visual para entender los ajustes de los modelos.


## Celda de Código 3

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(ar1_series)
plt.title('Serie Temporal Simulada AR(1)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor')
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 4 -->

### Comparación Visual de los Ajustes del Modelo

Ahora, vamos a superponer los valores ajustados por los modelos AR(1) y AR(2) sobre la serie original. Esto nos permitirá ver visualmente qué tan bien cada modelo replica el patrón de los datos.


## Celda de Código 5

```python
plt.figure(figsize=(14, 7))
plt.plot(ar1_series, label='Serie Original', color='blue')
plt.plot(results_ar1.predict(start=0, end=len(ar1_series)-1), label='Ajuste AR(1)', color='red', linestyle='--')
plt.plot(results_ar2.predict(start=0, end=len(ar1_series)-1), label='Ajuste AR(2)', color='green', linestyle=':')
plt.title('Serie Original vs. Ajustes de Modelos AR(1) y AR(2)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 6 -->

### Análisis de los Residuos del Modelo

Los residuos son la diferencia entre los valores observados y los valores predichos por el modelo. Un buen modelo debería tener residuos que se asemejen a ruido blanco (sin patrones). Graficar los residuos nos ayuda a evaluar la calidad del ajuste y a entender la 'información perdida' que AIC y BIC intentan cuantificar.


## Celda de Código 7

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Residuos del Modelo AR(1)
axes[0].plot(results_ar1.resid, color='red')
axes[0].set_title('Residuos del Modelo AR(1)')
axes[0].set_ylabel('Residuo')
axes[0].grid(True)

# Residuos del Modelo AR(2)
axes[1].plot(results_ar2.resid, color='green')
axes[1].set_title('Residuos del Modelo AR(2)')
axes[1].set_xlabel('Paso de Tiempo')
axes[1].set_ylabel('Residuo')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```


<!-- Markdown Cell 8 -->

### Interpretación de las Gráficas y su Relación con AIC/BIC

*   **Gráfica de Serie Temporal:** Muestra los datos que estamos intentando modelar.
*   **Comparación de Ajustes:** Idealmente, la línea de ajuste de un buen modelo debería seguir de cerca la serie original. Observaremos que el ajuste del AR(1) será muy similar al AR(2) visualmente, ya que la serie fue generada por un proceso AR(1).
*   **Gráfica de Residuos:** Los residuos del modelo AR(1) deberían ser más cercanos al ruido blanco, lo que indica que el modelo ha capturado la mayor parte de la estructura en los datos. Los residuos del AR(2) podrían mostrar un patrón similar pero no necesariamente mejor, ya que el parámetro adicional no es fundamental para la serie generada. Los valores de AIC y BIC formalizan esta observación visual, penalizando la complejidad adicional del AR(2) si no aporta una mejora significativa en el ajuste (es decir, en la reducción del ruido en los residuos).


<!-- Markdown Cell 9 -->

## Criterio de Información de Akaike (AIC)

El **Criterio de Información de Akaike (AIC)** es una medida de la calidad relativa de los modelos estadísticos para un conjunto de datos dado. Como tal, el AIC proporciona un medio para la selección del modelo. Dado un conjunto de modelos candidatos para los datos, el modelo preferido es aquel con el valor AIC más bajo.

El AIC se basa en la teoría de la información y estima la información perdida cuando se utiliza un modelo para representar el proceso que generó los datos. Se calcula como:

`AIC = 2k - 2ln(L)`

Donde:
*   `k` es el número de parámetros en el modelo.
*   `L` es el valor máximo de la función de verosimilitud (likelihood) del modelo.

Un valor de AIC más bajo indica un mejor ajuste del modelo, teniendo en cuenta la simplicidad del mismo. Penaliza la adición de parámetros (complejidad) para evitar el sobreajuste.


<!-- Markdown Cell 10 -->

## Criterio de Información Bayesiano (BIC)

El **Criterio de Información Bayesiano (BIC)**, también conocido como Criterio de Schwarz (SIC), es otro criterio para la selección de modelos entre un conjunto finito de modelos. Similar al AIC, busca el modelo con el valor BIC más bajo.

El BIC introduce una penalización más fuerte para el número de parámetros en el modelo que el AIC. Se calcula como:

`BIC = kln(n) - 2ln(L)`

Donde:
*   `k` es el número de parámetros en el modelo.
*   `n` es el número de observaciones (tamaño de la muestra).
*   `L` es el valor máximo de la función de verosimilitud del modelo.

Debido a la penalización `kln(n)`, que es más grande que `2k` (especialmente para `n` grandes), el BIC tiende a seleccionar modelos más parsimoniosos (con menos parámetros) que el AIC. Es útil cuando se busca un modelo que sea una buena representación del proceso de generación de datos, incluso si significa un ligero compromiso en el ajuste a los datos observados.


<!-- Markdown Cell 11 -->

## Ejemplo: Comparación de Modelos AR con AIC y BIC

Usaremos la serie temporal `ar1_series` que ya generamos, y ajustaremos un modelo autorregresivo (AR) de orden 1 y otro de orden 2 para comparar sus valores de AIC y BIC. El modelo con el AIC o BIC más bajo será considerado el "mejor" para nuestros datos.


## Celda de Código 12

```python

```


## Celda de Código 13

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np # Se añade import numpy para la generación de ar1_series

# --- Generar ar1_series --- (Copiar el código de la celda 64026f11 para asegurar su definición)
# Parameters for the AR(1) process
phi = 0.5  # Autoregressive coefficient (must be < 1 for stationarity)
c = 0    # Constant term
sigma = 1  # Standard deviation of the white noise
n = 200    # Number of observations

# Initialize the time series array
ar1_series = np.zeros(n)

# Generate white noise (innovations)
white_noise = np.random.normal(0, sigma, n)

# Simulate the AR(1) process
# X_t = c + phi * X_{t-1} + epsilon_t
for t in range(1, n):
    ar1_series[t] = c + phi * ar1_series[t-1] + white_noise[t]
# --- Fin de la generación de ar1_series ---


# --- Ajustar un modelo AR(1) ---
# El orden (p, d, q) para AR(p) es (p, 0, 0)
model_ar1 = ARIMA(ar1_series, order=(1, 0, 0))
results_ar1 = model_ar1.fit()

print("\n--- Resultados del Modelo AR(1) ---")
print(f"AIC: {results_ar1.aic:.2f}")
print(f"BIC: {results_ar1.bic:.2f}")

# --- Ajustar un modelo AR(2) ---
model_ar2 = ARIMA(ar1_series, order=(2, 0, 0))
results_ar2 = model_ar2.fit()

print("\n--- Resultados del Modelo AR(2) ---")
print(f"AIC: {results_ar2.aic:.2f}")
print(f"BIC: {results_ar2.bic:.2f}")


# --- Comparación ---
print("\n--- Comparación AIC/BIC ---")
if results_ar1.aic < results_ar2.aic:
    print(f"El modelo AR(1) tiene un AIC más bajo ({results_ar1.aic:.2f}) que el AR(2) ({results_ar2.aic:.2f}).")
else:
    print(f"El modelo AR(2) tiene un AIC más bajo ({results_ar2.aic:.2f}) que el AR(1) ({results_ar1.aic:.2f}).")

if results_ar1.bic < results_ar2.bic:
    print(f"El modelo AR(1) tiene un BIC más bajo ({results_ar1.bic:.2f}) que el AR(2) ({results_ar2.bic:.2f}).")
else:
    print(f"El modelo AR(2) tiene un BIC más bajo ({results_ar2.bic:.2f}) que el AR(1) ({results_ar1.bic:.2f}).")

```


<!-- Markdown Cell 14 -->

### Interpretación del Ejemplo

En este ejemplo, dado que la serie fue generada como un proceso AR(1) puro, esperamos que el modelo AR(1) tenga valores de AIC y BIC más bajos que el modelo AR(2). El AR(2) introduce un parámetro adicional que no mejora significativamente el ajuste del modelo, pero sí aumenta la penalización por complejidad, haciendo que sus valores de AIC y BIC sean probablemente más altos.

*   **AIC** selecciona el modelo que minimiza la pérdida de información.
*   **BIC** selecciona el modelo que tiene la mayor probabilidad posterior, suponiendo que el verdadero modelo está en el conjunto de modelos candidatos. Tiende a favorecer modelos más simples que el AIC, especialmente con grandes tamaños de muestra.


## Celda de Código 15

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the AR(1) process
phi = 0.5  # Autoregressive coefficient (must be < 1 for stationarity)
c = 0    # Constant term
sigma = 1  # Standard deviation of the white noise
n = 200    # Number of observations

# Initialize the time series array
ar1_series = np.zeros(n)

# Generate white noise (innovations)
white_noise = np.random.normal(0, sigma, n)

# Simulate the AR(1) process
# X_t = c + phi * X_{t-1} + epsilon_t
for t in range(1, n):
    ar1_series[t] = c + phi * ar1_series[t-1] + white_noise[t]

# Plot the simulated time series
plt.figure(figsize=(12, 6))
plt.plot(ar1_series)
plt.title(f'Simulated Stationary AR(1) Time Series (phi={phi})')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 16 -->

## Modelos ARIMA (AutoRegressive Integrated Moving Average)

Un modelo ARIMA es una generalización de los modelos de promedios móviles (MA) y autorregresivos (AR). Son ampliamente utilizados para la modelización de series temporales. La "I" en ARIMA significa "Integrado", lo que se refiere al uso de la diferenciación para hacer que la serie temporal sea estacionaria.

Un modelo ARIMA se especifica por 3 órdenes: `ARIMA(p, d, q)`:

*   **p (orden autorregresivo):** El número de observaciones pasadas que se incluirán en el modelo. Representa el número de términos AR.
*   **d (orden de diferenciación):** El número de veces que las observaciones brutas son diferenciadas para hacer que la serie temporal sea estacionaria. La diferenciación implica restar la observación actual de una observación pasada (por ejemplo, la observación anterior) para eliminar tendencias o estacionalidad.
*   **q (orden de media móvil):** El número de errores de predicción pasados que se incluirán en el modelo. Representa el número de términos MA.

**Estacionariedad:** Una serie temporal es estacionaria si sus propiedades estadísticas (media, varianza, autocorrelación) no cambian con el tiempo. Los modelos ARIMA asumen que la serie es estacionaria. Si la serie no es estacionaria, se aplica la diferenciación (`d > 0`) para transformarla en una serie estacionaria.


<!-- Markdown Cell 17 -->

## Funciones de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)

Las gráficas de ACF (Autocorrelation Function) y PACF (Partial Autocorrelation Function) son herramientas esenciales en el análisis de series temporales para identificar el orden apropiado de los componentes AR y MA en un modelo ARIMA.

### Función de Autocorrelación (ACF)

La **ACF** mide la correlación entre una serie temporal y sus versiones rezagadas en diferentes intervalos de tiempo. En una gráfica de ACF, el eje x representa el número de rezagos (lags) y el eje y representa el coeficiente de autocorrelación.

*   **AR(p) Model:** La ACF decae exponencialmente o con un patrón sinusoidal.
*   **MA(q) Model:** La ACF se corta abruptamente (cae a cero) después de `q` rezagos.

### Función de Autocorrelación Parcial (PACF)

La **PACF** mide la correlación entre una serie temporal y sus versiones rezagadas, pero eliminando la influencia de las correlaciones de los rezagos intermedios. Es decir, mide la correlación directa entre una observación y una observación rezagada.

*   **AR(p) Model:** La PACF se corta abruptamente (cae a cero) después de `p` rezagos.
*   **MA(q) Model:** La PACF decae exponencialmente o con un patrón sinusoidal.


<!-- Markdown Cell 18 -->

### Cálculo y Gráficas de ACF y PACF

Vamos a calcular y graficar la ACF y PACF para nuestra serie `ar1_series`. Esto nos ayudará a identificar los posibles órdenes `p` y `q` para un modelo ARIMA.


## Celda de Código 19

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

plt.subplot(121) # 1 fila, 2 columnas, primer gráfico
plot_acf(ar1_series, lags=30, ax=plt.gca(), title='Autocorrelación (ACF)')
plt.xlabel('Rezagos')
plt.ylabel('Autocorrelación')
plt.grid(True)
# Añadir marcador para Q (esperamos decaimiento gradual para AR(1), no un corte para Q)
# En un AR(1), la ACF decae exponencialmente, no se "corta" para un q.
# Podríamos marcar donde se vuelve no significativa o simplemente observar el decaimiento.
# Para fines ilustrativos de donde buscar un 'q' si fuera un MA, no hay un 'q' obvio aquí.
# Dejaremos la interpretación textual de la ACF para AR(1).

plt.subplot(122) # 1 fila, 2 columnas, segundo gráfico
plot_pacf(ar1_series, lags=30, ax=plt.gca(), title='Autocorrelación Parcial (PACF)')
plt.xlabel('Rezagos')
plt.ylabel('Autocorrelación Parcial')
plt.grid(True)
# Añadir marcador para P (esperamos un corte en el rezago 1 para AR(1))
plt.axvline(x=1, color='red', linestyle='--', label='P = 1 (corte significativo)')
plt.legend()

plt.tight_layout()
plt.show()
```


<!-- Markdown Cell 20 -->

### Interpretación de las Gráficas ACF y PACF

*   **En la gráfica de PACF:** Observa dónde la función de autocorrelación parcial se corta abruptamente (cae a cero) o se vuelve no significativa (dentro de las bandas de confianza). El número de rezagos antes de este corte sugiere el orden `p` del componente AR.
    *   Para una serie AR(1) como la `ar1_series` que generamos, esperamos que la PACF tenga un pico significativo en el rezago 1 y luego caiga a cero rápidamente.

*   **En la gráfica de ACF:** Observa dónde la función de autocorrelación se corta abruptamente. Este número de rezagos sugiere el orden `q` del componente MA.
    *   Para una serie AR(1) pura, la ACF debería decaer gradualmente (exponencialmente o con un patrón sinusoidal), no cortarse abruptamente.

*   **Para la diferenciación (d):** Si la ACF decae muy lentamente y es muy persistente en valores altos, podría indicar que la serie no es estacionaria y que se necesita diferenciación (`d > 0`). En nuestro caso, como la serie es generada AR(1) estacionaria, esperamos `d=0`.


<!-- Markdown Cell 21 -->

## Ajuste de Modelos ARIMA y Comparación de AIC/BIC

Ahora vamos a ajustar diferentes modelos ARIMA (iterando sobre posibles órdenes `p` y `q` y manteniendo `d=0` ya que nuestra serie es estacionaria) y calcularemos sus valores de AIC y BIC. Luego, graficaremos estos valores para visualizar qué combinaciones de órdenes (`p`, `q`) resultan en los modelos más adecuados según estos criterios.


## Celda de Código 22

```python

```


<!-- Markdown Cell 23 -->

### Revisión Visual de los Residuos de los Modelos Óptimos

Vamos a graficar los residuos de los dos modelos que fueron seleccionados como 'óptimos' según los criterios AIC y BIC. Como recordatorio:
*   El modelo **ARIMA(1,0,0)** fue el 'mejor' según **BIC**.
*   El modelo **ARIMA(1,0,1)** fue el 'mejor' según **AIC**.

Idealmente, los residuos de un buen modelo deben parecerse a un ruido blanco: distribuidos aleatoriamente alrededor de cero, sin patrones evidentes, autocorrelación significativa o cambios en la varianza.


## Celda de Código 24

```python
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# results_ar1 (ARIMA(1,0,0)) ya está disponible de celdas anteriores

# Ajustar el modelo ARIMA(1,0,1) si no se ha hecho explícitamente y sus resultados no están guardados
try:
    # Intentar usar results_best_aic si ya se definió en algún punto
    _ = results_best_aic.aic # Check if it exists
except NameError:
    # Si no existe, ajustarlo ahora
    model_best_aic = ARIMA(ar1_series, order=(1, 0, 1))
    results_best_aic = model_best_aic.fit()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Residuos del Modelo ARIMA(1,0,0) (Mejor según BIC)
axes[0].plot(results_ar1.resid, color='blue')
axes[0].set_title('Residuos del Modelo ARIMA(1,0,0) (Mejor según BIC)')
axes[0].set_ylabel('Residuo')
axes[0].grid(True)
axes[0].axhline(0, color='black', linestyle='--')

# Residuos del Modelo ARIMA(1,0,1) (Mejor según AIC)
axes[1].plot(results_best_aic.resid, color='green')
axes[1].set_title('Residuos del Modelo ARIMA(1,0,1) (Mejor según AIC)')
axes[1].set_xlabel('Paso de Tiempo')
axes[1].set_ylabel('Residuo')
axes[1].grid(True)
axes[1].axhline(0, color='black', linestyle='--')

plt.tight_layout()
plt.show()
```


<!-- Markdown Cell 25 -->

### Gráficos Q-Q de los Residuos del Modelo

Los gráficos Q-Q son una herramienta visual para evaluar si una muestra de datos (en este caso, nuestros residuos) sigue una distribución particular (generalmente la distribución normal). Si los puntos en el gráfico Q-Q se alinean aproximadamente a lo largo de una línea recta de 45 grados, sugiere que los residuos están distribuidos normalmente. Cualquier desviación de esta línea indica una desviación de la normalidad.


## Celda de Código 26

```python
import statsmodels.graphics.gofplots as smg
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Q-Q Plot para los Residuos del Modelo ARIMA(1,0,0) (Mejor según BIC)
smg.qqplot(results_ar1.resid, line='s', ax=axes[0])
axes[0].set_title('Q-Q Plot de Residuos ARIMA(1,0,0)')

# Q-Q Plot para los Residuos del Modelo ARIMA(1,0,1) (Mejor según AIC)
smg.qqplot(results_best_aic.resid, line='s', ax=axes[1])
axes[1].set_title('Q-Q Plot de Residuos ARIMA(1,0,1)')

plt.tight_layout()
plt.show()
```


<!-- Markdown Cell 27 -->

### Interpretación de los Gráficos Q-Q

*   **Línea de 45 grados (line='s'):** Esta línea diagonal representa la distribución normal teórica. Si los puntos de los residuos se agrupan estrechamente alrededor de esta línea, indica que los residuos siguen una distribución normal.
*   **Desviaciones:**
    *   Si los puntos se curvan en los extremos, sugiere colas más pesadas o más ligeras de lo esperado en una distribución normal.
    *   Si los puntos se desvían de la línea en el centro, puede indicar asimetría en la distribución de los residuos.

En nuestro caso, esperamos que los residuos se acerquen a una distribución normal, ya que el término de ruido blanco (`epsilon_t`) que utilizamos para generar la `ar1_series` fue muestreado de una distribución normal. Por lo tanto, los gráficos Q-Q deberían mostrar puntos alineados muy cerca de la línea de 45 grados, confirmando la suposición de normalidad de los errores.


<!-- Markdown Cell 28 -->

### Resumen de AIC y BIC para todos los Modelos ARIMA Evaluados

A continuación, se presenta un DataFrame que resume los valores de AIC y BIC para todas las combinaciones de órdenes ARIMA(p,0,q) que fueron evaluadas. Esto nos permite tener una visión general rápida de qué modelos tuvieron el mejor rendimiento según estos criterios.


## Celda de Código 29

```python
import pandas as pd

# Asegurarse de que 'results_ar1' y 'results_best_aic' estén disponibles
# Esto se maneja en la celda anterior donde se ajustan los modelos.

summary_df = pd.DataFrame({
    'Order (p,d,q)': model_orders,
    'AIC': aic_results,
    'BIC': bic_results
})

# Ordenar por AIC para ver los mejores modelos
print("\n--- Modelos ordenados por AIC ---")
display(summary_df.sort_values(by='AIC').reset_index(drop=True))

# Ordenar por BIC para ver los mejores modelos
print("\n--- Modelos ordenados por BIC ---")
display(summary_df.sort_values(by='BIC').reset_index(drop=True))
```


<!-- Markdown Cell 30 -->

## Generación de Predicciones con el Modelo ARIMA(1,0,0)

Ahora que hemos seleccionado y validado nuestro modelo ARIMA(1,0,0) como el más adecuado, podemos usarlo para generar predicciones futuras de la serie temporal. Es importante recordar que este modelo se basa en la estructura AR(1) de la serie, por lo que sus predicciones intentarán seguir esa dinámica.


## Celda de Código 31

```python
import matplotlib.pyplot as plt

# Definir el número de pasos de tiempo para la predicción
forecast_steps = 20 # Por ejemplo, predecir los próximos 20 pasos de tiempo

# Generar las predicciones
# El método get_forecast() permite obtener el intervalo de confianza.
forecast_result = results_ar1.get_forecast(steps=forecast_steps)

# Obtener los valores predichos y sus intervalos de confianza
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Visualizar la serie original, el ajuste del modelo y las predicciones
plt.figure(figsize=(16, 8))
plt.plot(ar1_series, label='Serie Original', color='blue')
plt.plot(results_ar1.predict(start=0, end=len(ar1_series)-1), label='Ajuste del Modelo ARIMA(1,0,0)', color='purple', linestyle='--')
plt.plot(range(len(ar1_series), len(ar1_series) + forecast_steps), forecast, label='Predicción ARIMA(1,0,0)', color='red')
plt.fill_between(range(len(ar1_series), len(ar1_series) + forecast_steps),
                 conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Intervalo de Confianza (95%)')

plt.title('Serie Original, Ajuste del Modelo y Predicciones ARIMA(1,0,0)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 32 -->

## 1. Carga y Preparación de Datos desde URL

Primero, descargaremos un conjunto de datos de temperaturas mínimas diarias en Melbourne, Australia, disponible públicamente. Este dataset es ideal para Prophet ya que tiene una columna de fecha y una columna de valor, y probablemente muestre estacionalidad.


## Celda de Código 33

```python
import pandas as pd
import matplotlib.pyplot as plt

# URL del dataset de temperaturas mínimas diarias
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
df = pd.read_csv(url, header=0)

# Mostrar las primeras filas y la información del DataFrame
print("Primeras 5 filas del dataset:")
display(df.head())
print("\nInformación del DataFrame:")
df.info()

# Prophet requiere que las columnas de fecha se llamen 'ds' y las de valor 'y'
# y que 'ds' sea de tipo datetime.

df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Temp': 'y'})

print("\nDataset preparado para Prophet (primeras 5 filas):")
display(df.head())

# Visualizar la serie temporal original
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'])
plt.title('Temperaturas Mínimas Diarias en Melbourne')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 34 -->

## 2. Instalación y Aplicación del Modelo Prophet

Ahora instalaremos la librería `prophet` (si aún no está instalada) y ajustaremos el modelo a nuestros datos. Prophet es robusto para manejar datos faltantes y cambios en la tendencia.


## Celda de Código 35

```python
# Instalar Prophet si es necesario
# !pip install prophet

from prophet import Prophet

# Crear una instancia del modelo Prophet
# Podemos añadir parámetros como la estacionalidad anual, semanal, etc.
model_prophet = Prophet(daily_seasonality=True)

# Ajustar el modelo a los datos
model_prophet.fit(df)

print("Modelo Prophet ajustado con éxito.")
```


<!-- Markdown Cell 36 -->

## 3. Generación de Predicciones con Prophet

Una vez que el modelo está ajustado, podemos generar un DataFrame con fechas futuras para realizar las predicciones. Vamos a predecir para el próximo año.


## Celda de Código 37

```python
# Crear un DataFrame con fechas futuras para la predicción (ej. 365 días más)
future = model_prophet.make_future_dataframe(periods=365)

# Realizar las predicciones
forecast = model_prophet.predict(future)

print("Predicciones generadas (primeras 5 filas del forecast):")
display(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
```


<!-- Markdown Cell 38 -->

## 4. Visualización de Predicciones y Componentes del Modelo Prophet

Prophet facilita la visualización de las predicciones, incluyendo el histórico, el pronóstico y los intervalos de confianza. También podemos ver la descomposición de la serie en sus componentes (tendencia, estacionalidad).


## Celda de Código 39

```python
# Gráfico de las predicciones
fig1 = model_prophet.plot(forecast)
plt.title('Predicción de Temperaturas Mínimas con Prophet')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.show()

# Gráfico de los componentes del modelo (tendencia, estacionalidad anual, semanal, diaria)
fig2 = model_prophet.plot_components(forecast)
plt.suptitle('Componentes del Modelo Prophet', y=1.02) # Ajustar título principal
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajustar layout para evitar solapamiento
plt.show()
```


<!-- Markdown Cell 40 -->

## 5. Nota sobre AIC y BIC con Prophet

Como mencionamos al principio, **Prophet no utiliza AIC o BIC directamente** para la selección del modelo o para reportar la bondad de ajuste en la misma forma que lo hacen los modelos estadísticos tradicionales como ARIMA.

Prophet se basa en un enfoque de modelado de series de tiempo con componentes additivos o multiplicativos (tendencia, estacionalidades y días festivos). La evaluación y selección de modelos en Prophet típicamente se realiza a través de:

*   **Validación cruzada:** Para evaluar el rendimiento predictivo del modelo en datos no vistos.
*   **Métricas de error:** Como RMSE (Error Cuadrático Medio), MAE (Error Absoluto Medio) o MAPE (Error Porcentual Absoluto Medio), que miden la precisión del pronóstico.
*   **Análisis visual:** De las predicciones y sus componentes, como las gráficas que acabamos de generar.

Por lo tanto, en el contexto de Prophet, la pregunta sobre un "buen modelo" se responde más por su rendimiento predictivo y la coherencia de sus componentes con las expectativas del dominio, más que por criterios de información como AIC o BIC.


<!-- Markdown Cell 41 -->

### Interpretación de la Gráfica de Predicción

La gráfica anterior muestra:

*   **Serie Original (azul):** Los datos que hemos analizado y modelado.
*   **Ajuste del Modelo (púrpura):** Cómo el modelo ARIMA(1,0,0) se ajusta a los datos históricos.
*   **Predicción (rojo):** Los valores futuros estimados por el modelo para los próximos `forecast_steps` (20 en este caso) pasos de tiempo.
*   **Intervalo de Confianza (rosa):** Una banda alrededor de la predicción que nos da una idea de la incertidumbre asociada a estas estimaciones. A medida que nos alejamos en el futuro, es normal que el intervalo de confianza se ensanche, reflejando el aumento de la incertidumbre.

Esta visualización es crucial para entender no solo las estimaciones puntuales del modelo, sino también la fiabilidad de dichas predicciones. Dado que la serie fue generada por un proceso AR(1) estacionario, esperamos que las predicciones tiendan a la media de la serie (que es cero en nuestro caso) a medida que el horizonte de predicción se extiende, y que la banda de confianza refleje esta convergencia y la incertidumbre inherente.


<!-- Markdown Cell 42 -->

### Detalle Completo de los Modelos "Óptimos"

Para un análisis a máximo detalle, presentamos los resúmenes completos de los modelos seleccionados como óptimos por AIC y BIC. Estas tablas incluyen coeficientes de los parámetros, errores estándar, valores p, y otras estadísticas importantes para evaluar la significancia y el ajuste del modelo.


## Celda de Código 43

```python
print("\n--- Resumen Detallado del Modelo ARIMA(1,0,0) (Mejor según BIC) ---")
display(results_ar1.summary())
```


## Celda de Código 44

```python
print("\n--- Resumen Detallado del Modelo ARIMA(1,0,1) (Mejor según AIC) ---")
display(results_best_aic.summary())
```


<!-- Markdown Cell 45 -->

### Interpretación de los Gráficos de Residuos

Al observar los gráficos de residuos, buscamos las siguientes características:

*   **Ruido Blanco:** Los residuos deben estar distribuidos aleatoriamente alrededor de cero. Esto significa que el modelo ha capturado la mayor parte de la estructura sistemática en la serie temporal.
*   **Sin Patrones:** No debería haber patrones visibles (tendencias, estacionalidad, agrupaciones de valores positivos/negativos). La presencia de patrones sugiere que el modelo no ha capturado completamente la información en los datos.
*   **Varianza Constante:** La dispersión de los residuos debe ser más o menos constante a lo largo del tiempo (homocedasticidad).
*   **Autocorrelación Mínima:** Idealmente, no debería haber autocorrelación significativa en los residuos, lo que se puede verificar con gráficas ACF/PACF de los propios residuos (no lo haremos aquí para simplificar, pero es un paso importante en un análisis real).

Dado que nuestra `ar1_series` fue generada por un proceso AR(1) puro, esperamos que los residuos del modelo ARIMA(1,0,0) se acerquen mucho al ruido blanco. Si el modelo ARIMA(1,0,1) muestra residuos muy similares, esto reforzaría la idea de que el término MA adicional no aportó mucha información nueva, confirmando la preferencia del BIC por el modelo más simple.


## Celda de Código 46

```python
import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # Ignorar warnings de convergencia

# Definir el rango de órdenes p y q a probar
p_values = range(0, 4) # p de 0 a 3
q_values = range(0, 4) # q de 0 a 3
d_value = 0 # Mantener d=0 ya que la serie es estacionaria

aic_results = []
bic_results = []
model_orders = []

for p in p_values:
    for q in q_values:
        order = (p, d_value, q)
        if p == 0 and q == 0:
            continue # Evitar el modelo (0,0,0) que no tiene parámetros
        try:
            model = ARIMA(ar1_series, order=order)
            model_fit = model.fit()
            aic_results.append(model_fit.aic)
            bic_results.append(model_fit.bic)
            model_orders.append(str(order))
            print(f'ARIMA{order} - AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}')
        except:
            # print(f'No se pudo ajustar ARIMA{order}')
            pass # Algunos modelos pueden no converger

# Encontrar el mejor modelo basado en AIC y BIC
best_aic_index = aic_results.index(min(aic_results))
best_bic_index = bic_results.index(min(bic_results))

print(f"\nMejor modelo según AIC: {model_orders[best_aic_index]} con AIC = {aic_results[best_aic_index]:.2f}")
print(f"Mejor modelo según BIC: {model_orders[best_bic_index]} con BIC = {bic_results[best_bic_index]:.2f}")

```


<!-- Markdown Cell 47 -->

### Gráficas de AIC y BIC por Orden de Modelo

Visualicemos los valores de AIC y BIC para cada modelo ajustado. Esto nos permitirá ver claramente qué órdenes `(p, q)` minimizan estos criterios.


## Celda de Código 48

```python
plt.figure(figsize=(16, 8))

plt.plot(model_orders, aic_results, marker='o', linestyle='-', color='blue', label='AIC')
plt.plot(model_orders, bic_results, marker='x', linestyle='--', color='red', label='BIC')

plt.title('Comparación de AIC y BIC para Diferentes Órdenes ARIMA(p,0,q)')
plt.xlabel('Orden del Modelo (p,0,q)')
plt.ylabel('Valor del Criterio')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```


<!-- Markdown Cell 49 -->

### Identificación de P y Q a partir de las Gráficas ACF y PACF

Las gráficas de Autocorrelación (ACF) y Autocorrelación Parcial (PACF) son herramientas visuales fundamentales para determinar los órdenes `p` (Autorregresivo) y `q` (Media Móvil) de un modelo ARIMA.

Para nuestra serie `ar1_series`, que fue generada por un proceso AR(1) ($X_t = c + \phi X_{t-1} + \epsilon_t$), esperábamos un comportamiento específico:

1.  **Determinación de `P` (Orden del Componente Autorregresivo):**
    *   Observamos la **gráfica de PACF (Autocorrelación Parcial)**. Para un proceso AR(p), la PACF se "corta" o cae abruptamente a cero (o dentro de las bandas de confianza) después del rezago `p`.
    *   En nuestra gráfica de PACF, vemos un **pico significativo en el rezago 1**, y luego las autocorrelaciones parciales caen rápidamente a valores no significativos (dentro de las bandas azules de confianza).
    *   Esto indica claramente que **P = 1**, lo cual es consistente con nuestra serie AR(1).

2.  **Determinación de `Q` (Orden del Componente de Media Móvil):**
    *   Observamos la **gráfica de ACF (Autocorrelación)**. Para un proceso MA(q), la ACF se "corta" o cae abruptamente a cero (o dentro de las bandas de confianza) después del rezago `q`.
    *   Sin embargo, para un proceso AR(p) puro (como nuestra `ar1_series`), la ACF no se "corta" abruptamente; en su lugar, **decae gradualmente**, a menudo de forma exponencial o sinusoidal, a medida que los rezagos aumentan.
    *   En nuestra gráfica de ACF, observamos un decaimiento gradual, lo que sugiere que **Q = 0** si el proceso es un AR puro. Si hubiera un "corte" abrupto en un rezago específico, ese sería nuestro `q`.


<!-- Markdown Cell 50 -->

### Interpretación de las Gráficas de AIC y BIC

Al observar estas gráficas, el modelo con el punto más bajo en la línea de AIC o BIC es el que se considera óptimo según ese criterio. Es común que BIC favorezca modelos más simples que AIC, especialmente con conjuntos de datos grandes, debido a su penalización más fuerte por la complejidad del modelo.

En el caso de nuestra `ar1_series` que fue generada por un proceso AR(1), esperaríamos que el modelo ARIMA(1,0,0) (que es equivalente a un AR(1)) tuviera los valores más bajos de AIC y BIC, o al menos que BIC lo favoreciera por su simplicidad.


<!-- Markdown Cell 51 -->

### Visualización del Ajuste del Mejor Modelo ARIMA

Basándonos en el análisis de AIC y BIC, especialmente considerando que la serie fue generada por un proceso AR(1), el modelo ARIMA(1,0,0) es el más parsimonioso y adecuado según el BIC. Vamos a graficar su ajuste sobre la serie original para ver visualmente cómo se comporta.


## Celda de Código 52

```python
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# El modelo AR(1) ya fue ajustado y está disponible como results_ar1, que es ARIMA(1,0,0)
# Si quisieras usar el mejor AIC (1,0,1) o cualquier otro, lo ajustarías aquí:
# model_best_aic = ARIMA(ar1_series, order=(1, 0, 1))
# results_best_aic = model_best_aic.fit()

plt.figure(figsize=(14, 7))
plt.plot(ar1_series, label='Serie Original', color='blue')
plt.plot(results_ar1.predict(start=0, end=len(ar1_series)-1), label='Ajuste ARIMA(1,0,0)', color='purple', linestyle='-')
plt.title('Serie Original vs. Ajuste del Modelo ARIMA(1,0,0)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()
```


<!-- Markdown Cell 53 -->

### Interpretación de la Gráfica del Modelo ARIMA

Esta gráfica nos muestra cómo el modelo ARIMA(1,0,0) (que es equivalente a un modelo AR(1)) se ajusta a la serie `ar1_series` original. Deberías observar que la línea de ajuste sigue de cerca los patrones de la serie original, lo que indica un buen rendimiento del modelo para capturar la dinámica de los datos. Esta visualización complementa la información proporcionada por los criterios AIC y BIC, confirmando que el modelo seleccionado es una representación adecuada de la serie temporal.


<!-- Markdown Cell 54 -->

## Conclusión General del Análisis de Modelos AR y ARIMA

En este análisis hemos explorado los fundamentos de la modelización de series temporales, centrándonos en los modelos Autorregresivos (AR) y ARIMA, y cómo se utilizan los Criterios de Información de Akaike (AIC) y Bayesiano (BIC) para la selección de modelos.

**1. Generación y Visualización de la Serie Temporal:**

Simulamos una serie `ar1_series` a partir de un proceso AR(1), lo que nos dio un punto de referencia claro para evaluar el rendimiento de nuestros modelos.

**2. Comparación de Modelos AR con AIC y BIC:**

Ajustamos un modelo AR(1) y un AR(2) a la serie generada. Observamos que:
*   El **AIC** favoreció ligeramente al modelo AR(2), lo que sugiere que pudo haber capturado alguna pequeña variación adicional.
*   El **BIC**, con su penalización más estricta por complejidad, seleccionó el modelo AR(1) como el mejor. Esto es coherente con la naturaleza de la serie generada y subraya la preferencia del BIC por modelos más parsimoniosos cuando la mejora de ajuste no justifica la complejidad adicional.

Las gráficas de ajuste visual confirmaron que ambos modelos replicaron bastante bien la serie original, mientras que el análisis de residuos mostró qué tan cerca cada modelo se acercó a un comportamiento de "ruido blanco".

**3. Modelos ARIMA y Funciones ACF/PACF:**

Repasamos la teoría detrás de los modelos ARIMA y la importancia de las Funciones de Autocorrelación (ACF) y Autocorrelación Parcial (PACF) para determinar los órdenes `p` (AR) y `q` (MA).:
*   En la **gráfica PACF** de nuestra `ar1_series`, el pico significativo en el **rezago 1** y la caída abrupta posterior claramente señalaron un orden `p=1`.
*   En la **gráfica ACF**, el decaimiento gradual (sin un corte abrupto) es consistente con un modelo AR(1) puro, lo que sugiere un orden `q=0`.

**4. Ajuste de Modelos ARIMA e Interpretación Gráfica de AIC/BIC:**

Iteramos sobre diferentes combinaciones de órdenes `(p, 0, q)` para los modelos ARIMA y calculamos sus valores de AIC y BIC. La gráfica de estos criterios nos permitió visualizar que el modelo **ARIMA(1,0,0)** (equivalente a un AR(1)) fue el más consistente con los resultados del BIC, y tuvo un AIC competitivo, lo que nuevamente reforzó que el proceso AR(1) era el subyacente.

**5. Visualización del Ajuste del Mejor Modelo ARIMA:**

Finalmente, graficamos el ajuste del modelo ARIMA(1,0,0) sobre la serie original. Esta visualización demostró que el modelo seleccionado es una representación robusta y precisa de la dinámica de nuestra serie temporal simulada.

En resumen, este ejercicio ha ilustrado cómo el AIC y el BIC, junto con las herramientas visuales como las gráficas de ACF, PACF y los ajustes de modelos, son fundamentales para la identificación y selección del modelo más apropiado en el análisis de series temporales.


## Celda de Código 55

```python
import numpy as np
import matplotlib.pyplot as plt

# Calculate ZT using the formula ZT = XT - 0.5 * XT-1
# Note: XT is represented by ar1_series

# Initialize ZT series, it will have n-1 elements if starting from t=1
zt_series = np.zeros(n)

# ZT[0] cannot be calculated as XT[-1] is not defined in this context.
# Let's assume ZT starts from the second element, corresponding to XT[1].
# Alternatively, we can define ZT[0] based on XT[0] if XT[-1] is assumed 0 or use the first calculated ZT.
# For simplicity, let's calculate from t=1 to n-1
# In the context of the AR(1) process X_t = c + phi * X_{t-1} + epsilon_t,
# if phi = 0.5, then epsilon_t = X_t - 0.5 * X_{t-1}. So ZT is essentially the white noise term.

# Let's calculate ZT as epsilon_t directly from the formula given.
# We already have white_noise which is epsilon_t.
zt_series = white_noise

# Plot the simulated ZT series (which is the white noise / innovations)
plt.figure(figsize=(12, 6))
plt.plot(zt_series)
plt.title('Simulated ZT Series (Innovations)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```


## Celda de Código 56

```python
ventas = pd.read_csv('california_housing_train.csv')
ventas.head(101)
```


## Celda de Código 57

```python
pltr = yf.download("BK", start="1900-01-01")
pltr
```


## Celda de Código 58

```python
pltr_price=pltr['Close']
pltr_price
```


## Celda de Código 59

```python
import plotly.express as px

fig = px.line(pltr_price, x=pltr_price.index, y='PLTR', title='PLTR Close Price Over Time')
fig.show()
```


## Celda de Código 60

```python

```


<!-- Markdown Cell 61 -->

## Modelo SARIMA (Seasonal ARIMA)

El modelo SARIMA (Seasonal AutoRegressive Integrated Moving Average) extiende el modelo ARIMA para incluir componentes estacionales. Se especifica como `SARIMA(p, d, q)(P, D, Q)s`, donde:

*   `(p, d, q)` son las órdenes no estacionales (AR, I, MA).
*   `(P, D, Q)` son las órdenes estacionales (AR, I, MA).
*   `s` es la longitud del período estacional (por ejemplo, 7 para datos diarios con estacionalidad semanal, 12 para datos mensuales con estacionalidad anual).

Para este ejercicio, ajustaremos un modelo SARIMA(0,1,1)(0,1,1) con un período estacional `s=7` a la serie de temperaturas mínimas diarias de Melbourne, lo que implica:
*   No hay componente AR no estacional (`p=0`).
*   Una diferenciación no estacional (`d=1`), para manejar tendencias.
*   Un componente MA no estacional (`q=1`).
*   No hay componente AR estacional (`P=0`).
*   Una diferenciación estacional (`D=1`), para manejar estacionalidad.
*   Un componente MA estacional (`Q=1`).
*   Un período estacional de 7 días (`s=7`), asumiendo una estacionalidad semanal en las temperaturas diarias.


## Celda de Código 62

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse de que el DataFrame 'df' de temperaturas esté disponible y con el formato correcto
# Ya fue preparado en la celda 0ee811c0
# df['ds'] = pd.to_datetime(df['ds'])
# df = df.set_index('ds')

# Definir los órdenes del modelo SARIMA
order = (0, 1, 1)        # (p, d, q) no estacional
seasonal_order = (0, 1, 1, 7) # (P, D, Q, s) estacional, con s=7 para estacionalidad semanal

print(f"Ajustando modelo SARIMA{order}{seasonal_order}...")

# Ajustar el modelo SARIMA
try:
    # Usamos la serie 'y' que contiene las temperaturas
    model_sarima = sm.tsa.SARIMAX(df['y'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results_sarima = model_sarima.fit(disp=False) # disp=False para evitar la salida de convergencia
    print("Modelo SARIMA ajustado con éxito.")
    print(results_sarima.summary())
except Exception as e:
    print(f"Error al ajustar el modelo SARIMA: {e}")
```


## Celda de Código 63

```python
import matplotlib.pyplot as plt

# Generar predicciones para los datos históricos
# start y end se refieren a las posiciones en el índice, no a las fechas.
# La serie original df['y'] tiene un índice de 0 a n-1, si no se seteó el índice de fecha.
# Asumiendo que df['y'] es una serie con un índice DatetimeIndex

# Obtener las fechas del DataFrame original para el plot
original_dates = df['ds']

# Predicciones in-sample (ajuste del modelo a los datos históricos)
# Usamos el índice de `df` que ya tiene la columna 'ds' como datetime
start_index_fit = 0
end_index_fit = len(df['y']) - 1
# Las predicciones se alinearán con el índice de tiempo de la serie original.
predictions_sarima = results_sarima.predict(start=start_index_fit, end=end_index_fit, dynamic=False)

# Generar pronósticos para el futuro (ej. los próximos 30 días)
forecast_steps = 30
forecast_sarima_result = results_sarima.get_forecast(steps=forecast_steps)
forecast_sarima = forecast_sarima_result.predicted_mean
conf_int_sarima = forecast_sarima_result.conf_int()

# Crear un índice de fechas para el pronóstico
last_date_original = original_dates.max()
forecast_dates = pd.date_range(start=last_date_original + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

plt.figure(figsize=(18, 8))
plt.plot(original_dates, df['y'], label='Serie Original', color='blue')
plt.plot(original_dates, predictions_sarima, label='Ajuste SARIMA', color='green', linestyle='--')
plt.plot(forecast_dates, forecast_sarima, label=f'Pronóstico SARIMA ({forecast_steps} días)', color='red')
plt.fill_between(forecast_dates,
                 conf_int_sarima.iloc[:, 0],
                 conf_int_sarima.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo de Confianza (95%)')

plt.title(f'Serie Original, Ajuste y Pronóstico con SARIMA{order}{seasonal_order}')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Pronósticos Futuros del Modelo SARIMA ---")
# Combine forecast dates with forecast values and confidence intervals for display
forecast_df = pd.DataFrame({
    'Fecha': forecast_dates,
    'Pronóstico': forecast_sarima.values,
    'Límite Inferior (95%)': conf_int_sarima.iloc[:, 0].values,
    'Límite Superior (95%)': conf_int_sarima.iloc[:, 1].values
})
display(forecast_df)
```


<!-- Markdown Cell 64 -->

## Búsqueda del Modelo SARIMA Óptimo usando AIC y BIC

Para encontrar el modelo SARIMA más adecuado para nuestras temperaturas, vamos a realizar una búsqueda exhaustiva (grid search) a través de diferentes combinaciones de órdenes no estacionales `(p, d, q)` y estacionales `(P, D, Q, s)`. Evaluaremos cada modelo utilizando los criterios AIC (Akaike Information Criterion) y BIC (Bayesian Information Criterion). El modelo con los valores más bajos de AIC y/o BIC será considerado el 'óptimo', ya que estos criterios penalizan la complejidad del modelo al mismo tiempo que evalúan su ajuste a los datos.

Mantendremos `d=1` y `D=1` para la diferenciación no estacional y estacional respectivamente, ya que la serie de temperaturas mínimas diarias suele presentar tendencia y estacionalidad. El período estacional `s` se fijará en `7` (semanal).


## Celda de Código 65

```python
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # Ignorar warnings de convergencia

# Definir los rangos de órdenes p, q, P, Q a probar
p_values = range(0, 2)  # p de 0 a 1
q_values = range(0, 2)  # q de 0 a 1
P_values = range(0, 2)  # P de 0 a 1
Q_values = range(0, 2)  # Q de 0 a 1
d_value = 1             # Diferenciación no estacional (fijo)
D_value = 1             # Diferenciación estacional (fijo)
s_value = 7             # Período estacional (fijo para semanal)

aic_results_sarima = []
bic_results_sarima = []
model_orders_sarima = []

# Asegurarse de que el DataFrame 'df' de temperaturas esté disponible y con el formato correcto
# Ya fue preparado en la celda 0ee811c0
# Establecer la columna 'ds' como índice si no lo está para SARIMAX
# sarimax_df = df.set_index('ds')

print("Iniciando búsqueda de modelos SARIMA...")

for p in p_values:
    for q in q_values:
        for P in P_values:
            for Q in Q_values:
                order = (p, d_value, q)
                seasonal_order = (P, D_value, Q, s_value)

                if p == 0 and q == 0 and P == 0 and Q == 0: # Evitar el modelo (0,1,0)(0,1,0)7 que no tiene parámetros
                    continue

                try:
                    model = SARIMAX(df['y'], order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False) # disp=False para evitar la salida de convergencia
                    aic_results_sarima.append(model_fit.aic)
                    bic_results_sarima.append(model_fit.bic)
                    model_orders_sarima.append(f'SARIMA{order}{seasonal_order}')
                    print(f'SARIMA{order}{seasonal_order} - AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}')
                except Exception as e:
                    # print(f'No se pudo ajustar SARIMA{order}{seasonal_order}: {e}')
                    pass # Algunos modelos pueden no converger o dar errores

# Encontrar el mejor modelo basado en AIC y BIC
if aic_results_sarima and bic_results_sarima:
    best_aic_index_sarima = aic_results_sarima.index(min(aic_results_sarima))
    best_bic_index_sarima = bic_results_sarima.index(min(bic_results_sarima))

    print(f"\nMejor modelo según AIC: {model_orders_sarima[best_aic_index_sarima]} con AIC = {aic_results_sarima[best_aic_index_sarima]:.2f}")
    print(f"Mejor modelo según BIC: {model_orders_sarima[best_bic_index_sarima]} con BIC = {bic_results_sarima[best_bic_index_sarima]:.2f}")
else:
    print("No se pudieron ajustar modelos SARIMA válidos con los parámetros seleccionados.")

# Crear un DataFrame para un resumen de los resultados
summary_sarima_df = pd.DataFrame({
    'Order': model_orders_sarima,
    'AIC': aic_results_sarima,
    'BIC': bic_results_sarima
})

print("\n--- Resumen de Modelos SARIMA Ordenados por AIC ---")
display(summary_sarima_df.sort_values(by='AIC').reset_index(drop=True))

print("\n--- Resumen de Modelos SARIMA Ordenados por BIC ---")
display(summary_sarima_df.sort_values(by='BIC').reset_index(drop=True))
```


## Celda de Código 66

```python
plt.figure(figsize=(16, 8))

plt.plot(model_orders_sarima, aic_results_sarima, marker='o', linestyle='-', color='blue', label='AIC')
plt.plot(model_orders_sarima, bic_results_sarima, marker='x', linestyle='--', color='red', label='BIC')

plt.title('Comparación de AIC y BIC para Diferentes Órdenes SARIMA')
plt.xlabel('Orden del Modelo (p,d,q)(P,D,Q,s)')
plt.ylabel('Valor del Criterio')
plt.xticks(rotation=90, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


<!-- Markdown Cell 67 -->

### Visualización del Ajuste y Pronóstico del Modelo SARIMA Óptimo

Una vez identificados los modelos SARIMA óptimos según AIC y BIC, vamos a visualizar el ajuste del mejor modelo (considerando el equilibrio entre ajuste y parsimonia) a los datos históricos y sus pronósticos futuros. En muchos casos, el modelo con el BIC más bajo es preferible por su mayor penalización a la complejidad.


## Celda de Código 68

```python
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import re # Para extraer los órdenes del string

# Obtener el mejor modelo según BIC (que tiende a ser más parsimonioso)
best_bic_order_str = model_orders_sarima[best_bic_index_sarima]

# Extraer los órdenes (p,d,q) y (P,D,Q,s) del string
match = re.match(r'SARIMA\((\d),\s*(\d),\s*(\d)\)\((\d),\s*(\d),\s*(\d),\s*(\d)\)', best_bic_order_str)

if match:
    p_opt, d_opt, q_opt, P_opt, D_opt, Q_opt, s_opt = map(int, match.groups())
    optimal_order = (p_opt, d_opt, q_opt)
    optimal_seasonal_order = (P_opt, D_opt, Q_opt, s_opt)
else:
    print(f"No se pudieron extraer los órdenes del string: {best_bic_order_str}. Usando valores predeterminados.")
    optimal_order = (0, 1, 1)
    optimal_seasonal_order = (0, 1, 1, 7)

print(f"Ajustando el modelo SARIMA óptimo: SARIMA{optimal_order}{optimal_seasonal_order}")

# Ajustar el modelo SARIMA óptimo
model_optimal_sarima = SARIMAX(df['y'], order=optimal_order, seasonal_order=optimal_seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
results_optimal_sarima = model_optimal_sarima.fit(disp=False)

# Obtener las fechas del DataFrame original para el plot
original_dates = df['ds']

# Predicciones in-sample (ajuste del modelo a los datos históricos)
start_index_fit = 0
end_index_fit = len(df['y']) - 1
predictions_optimal_sarima = results_optimal_sarima.predict(start=start_index_fit, end=end_index_fit, dynamic=False)

# Generar pronósticos para el futuro (ej. los próximos 30 días)
forecast_steps_optimal = 30
forecast_optimal_sarima_result = results_optimal_sarima.get_forecast(steps=forecast_steps_optimal)
forecast_optimal_sarima = forecast_optimal_sarima_result.predicted_mean
conf_int_optimal_sarima = forecast_optimal_sarima_result.conf_int()

# Crear un índice de fechas para el pronóstico
last_date_original = original_dates.max()
forecast_dates_optimal = pd.date_range(start=last_date_original + pd.Timedelta(days=1), periods=forecast_steps_optimal, freq='D')

plt.figure(figsize=(18, 8))
plt.plot(original_dates, df['y'], label='Serie Original', color='blue')
plt.plot(original_dates, predictions_optimal_sarima, label=f'Ajuste SARIMA{optimal_order}{optimal_seasonal_order}', color='green', linestyle='--')
plt.plot(forecast_dates_optimal, forecast_optimal_sarima, label=f'Pronóstico SARIMA{optimal_order}{optimal_seasonal_order}', color='red')
plt.fill_between(forecast_dates_optimal,
                 conf_int_optimal_sarima.iloc[:, 0],
                 conf_int_optimal_sarima.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo de Confianza (95%)')

plt.title(f'Serie Original, Ajuste y Pronóstico con SARIMA{optimal_order}{optimal_seasonal_order}')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Pronósticos Futuros del Modelo SARIMA Óptimo ---")
# Combine forecast dates with forecast values and confidence intervals for display
forecast_optimal_df = pd.DataFrame({
    'Fecha': forecast_dates_optimal,
    'Pronóstico': forecast_optimal_sarima.values,
    'Límite Inferior (95%)': conf_int_optimal_sarima.iloc[:, 0].values,
    'Límite Superior (95%)': conf_int_optimal_sarima.iloc[:, 1].values
})
display(forecast_optimal_df)

```
