# Documentación del Notebook: Ruido Blanco


## Código 1

```python
import numpy as np
import matplotlib.pyplot as plt

# número de observaciones
n = 200

# generar ruido blanco
Z = np.random.normal(0, 1, n)

# inicializar serie
X = np.zeros(n)

# generar proceso AR(2)
for t in range(2, n):
    X[t] = (1/3)*X[t-1] + (1/2)*X[t-2] + Z[t]

# graficar
plt.figure(figsize=(10,5))
plt.plot(X)
plt.title(r"Simulación $X_t = \frac{1}{3}X_{t-1} + \frac{1}{2}X_{t-2} + Z_t$")
plt.xlabel("t")
plt.ylabel("X_t")
plt.grid(True)

plt.show()
```


## Código 2

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 200
Z = np.random.normal(0,1,T)

X = np.zeros(T)

for t in range(2, T):
    X[t] = (1/3)*X[t-1] + (1/2)*X[t-2] + Z[t]

plt.figure(figsize=(10,5))
plt.plot(X, linewidth=2)
plt.title(r"Simulación del proceso $X_t - \frac{1}{3}X_{t-1} - \frac{1}{2}X_{t-2} = Z_t$")
plt.xlabel("t")
plt.ylabel("$X_t$")
plt.grid()

plt.show()

```
