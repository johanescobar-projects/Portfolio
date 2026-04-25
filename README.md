# SMOTE: Formalización Matemática y Validación Empírica del Sobremuestreo Sintético para el Desbalance de Clases en Machine Learning

> **Estudio aplicado sobre SMOTE, Borderline-SMOTE y ADASYN en clasificación desbalanceada con MLP.**

**Autor:** Johan Escobar
**Institución:** Fundación Universitaria Los Libertadores — Bogotá, Colombia  
**Semillero:** SIMODEST | **Grupo:** GIDAD  

---

##  Resumen

Este proyecto evalúa el efecto de **tres variantes de SMOTE** sobre el desempeño de una **red neuronal multicapa (MLP)** en escenarios simulados con desbalance de clases y estructura estadística controlada.

El objetivo **no** es maximizar métricas, sino **aislar y medir el efecto de cada técnica de remuestreo** manteniendo constantes el modelo, la división de datos, el protocolo experimental y las métricas de evaluación.

---


## Objetivo

Evaluar el efecto de tres variantes de SMOTE sobre el rendimiento de una red neuronal multicapa (MLP) en datos simulados con estructura estadística realista, mediante un protocolo experimental reproducible y análisis estadístico de los resultados.

> El objetivo **no** es optimizar métricas. Es aislar y medir el efecto de cada variante de remuestreo manteniendo todo lo demás constante.

---

## Instalación


**Python:** 3.8+  
**Dependencias principales:** `scikit-learn`, `imbalanced-learn`, `numpy`, `pandas`, `matplotlib`

---

## Diseño experimental

### Escenarios simulados

| Escenario | Tasa clase minoritaria | Ratio | Característica diferenciadora |
|---|---|---|---|
| Riesgo crediticio | ~7 % | 1:13 | Solapamiento moderado (σ = 0.6) |
| Deserción escolar | ~8 % | 1:11 | Alta varianza interna en la minoría (σ = 1.2) |
| Fraude bancario | ~6 % | 1:15 | 20 % de fraudes camuflados con perfil similar a legítimos |

Los datos se generan mediante **funciones logísticas con ruido gaussiano**. Esta es una decisión metodológica deliberada, no una limitación: permite controlar el nivel de solapamiento y el ratio de desbalance de forma independiente, algo imposible con datos reales. Esta estrategia sigue el marco ADEMP de Morris, White & Crowther (2019).

### Variable experimental

Solo varía el **método de remuestreo**:

- Sin remuestreo (línea base)
- SMOTE clásico — Chawla et al. (2002)
- Borderline-SMOTE — Han, Wang & Mao (2005)
- ADASYN — He, Bai, Garcia & Li (2008)

Todo lo demás permanece constante: arquitectura, hiperparámetros, semilla, división.

### Arquitectura de la red

**Arquitectura fija:** `(32,16,8)` — dos capas ocultas con activación ReLU, optimizador Adam, early stopping.

Los hiperparámetros de optimización varían por escenario (fijados antes de los experimentos):

| Parámetro | Crédito | Deserción | Fraude |
|---|---|---|---|
| `learning_rate_init` | 0.0005 | 0.001 | 0.0005 |
| `alpha` (L2) | 0.0001 | 0.0001 | 0.001 |
| `n_iter_no_change` | 15 | 10 | 20 |
| `max_iter` | 400 | 300 | 500 |

### Protocolo (reglas de oro)

1. División estratificada 70 % train / 30 % test con `stratify=y`
2. `StandardScaler` ajustado **solo** sobre entrenamiento
3. SMOTE aplicado **solo** sobre el entrenamiento escalado
4. Evaluación sobre prueba **original** sin modificar
5. Semilla global: `SEED = 12345`
6. Análisis multi-semilla con 5 semillas: `[12345, 22222, 33333, 44444, 55555]`

---

## Métricas

| Métrica | Rol |
|---|---|
| **PR-AUC** | Métrica principal — no se ve distorsionada por el desbalance de clases |
| **ROC-AUC** | Métrica de referencia — complementaria |
| **F1-Score** | Balance precisión/recall con umbral de Youden |
| **Sensibilidad (Recall)** | Capacidad de detectar la clase minoritaria |
| **G-mean** | sqrt(Sensibilidad × Especificidad) — complementa F1 sin depender del umbral |

El **umbral de Youden** (`J = Sensibilidad + Especificidad - 1`) reemplaza el umbral 0.5 por defecto, ya que con datos desbalanceados la red tiende a clasificar todo como clase 0.

---

## Resultados obtenidos — Crédito (3 semillas, umbral de Youden)

| Método | F1 | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Sin remuestreo | 0.136 ± 0.004 | 0.844 ± 0.164 | 0.070 ± 0.008 | 0.483 ± 0.047 |
| SMOTE | 0.175 ± 0.018 | 0.712 ± 0.006 | 0.117 ± 0.008 | 0.630 ± 0.033 |
| Borderline-SMOTE | 0.189 ± 0.020 | 0.632 ± 0.117 | 0.118 ± 0.009 | 0.642 ± 0.046 |
| ADASYN | 0.182 ± 0.027 | 0.631 ± 0.097 | 0.108 ± 0.020 | 0.614 ± 0.055 |

Las métricas bajas son **esperadas y metodológicamente correctas** con datos solapados. El objetivo es medir la mejora relativa de SMOTE, no alcanzar valores absolutos altos.

**Rango realista esperable:**
- Crédito: F1 ∈ [0.15, 0.35] | Recall ∈ [0.50, 0.75] | PR-AUC ∈ [0.10, 0.25]
- Fraude: F1 ∈ [0.20, 0.35] | Recall ∈ [0.50, 0.70] | PR-AUC ∈ [0.20, 0.35]

---

## Proposiciones matemáticas verificadas

### Proposición 1 — Valor esperado del punto sintético

SMOTE genera: `x_nuevo = x_i + λ·(x_vecino − x_i)` con `λ ~ Uniforme(0,1)`

Como `E[λ] = 0.5`:  `E[x_nuevo] = (x_i + x_vecino) / 2`

Verificado con 20,000 pares: diferencia < 0.01 en todas las dimensiones. ✓

### Proposición 2 — Separación y dispersión de sintéticos

A mayor separación entre clases, mayor desplazamiento de la media local tras SMOTE. En escenarios solapados (los tres del proyecto), los sintéticos caen en la zona de confusión, lo que explica por qué SMOTE no siempre mejora en datos realistas.

---

## Por qué no se usa `sample_weight`

Los experimentos preliminares mostraron que `sample_weight` supera a SMOTE en todos los escenarios. Sin embargo, **se excluye deliberadamente**: mezclarlo haría imposible aislar el efecto de SMOTE, que es el objetivo del estudio.

---

## Análisis de sensibilidad a k (deserción)

Se varió `k ∈ {3, 5, 7, 10}`. Resultado: k = 7 da el mejor ROC-AUC (0.701), k = 5 el mejor Recall (0.581). Esto se conecta con la Proposición 2: con k grande y alta varianza interna, los vecinos pueden pertenecer a subgrupos distintos de desertores.

---

## Literatura citada

| # | Referencia | DOI | Función |
|---|---|---|---|
| 1 | Morris, White & Crowther (2019). *Statistics in Medicine, 38*, 2074–2102. | [10.1002/sim.8086](https://doi.org/10.1002/sim.8086) | Justificación de datos simulados — marco ADEMP |
| 2 | de Zarzà, de Curtò & Calafate (2023). *Electronics, 12*, 2674. | [10.3390/electronics12122674](https://doi.org/10.3390/electronics12122674) | Protocolo MLP + métricas |
| 3 | Chawla et al. (2002). *JAIR, 16*, 321–357. | [10.1613/jair.953](https://doi.org/10.1613/jair.953) | SMOTE clásico — técnica base |
| 4 | Mujahid et al. (2024). *Journal of Big Data, 11*, 87. | [10.1186/s40537-024-00943-4](https://doi.org/10.1186/s40537-024-00943-4) | Revisión de las 3 variantes |
| 5 | Han, Wang & Mao (2005). *Advances in Intelligent Computing*, 878–887. | [10.1007/11538059_91](https://doi.org/10.1007/11538059_91) | Borderline-SMOTE |
| 6 | He, Bai, Garcia & Li (2008). *IJCNN*, 1322–1328. | [10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969) | ADASYN |
| 7 | Yang, Wang, Shi & Qiu (2024). *BDCC, 8*, 151. | [10.3390/bdcc8110151](https://doi.org/10.3390/bdcc8110151) | Contexto fraude (opcional) |
| 8 | Lemaitre, Nogueira & Aridas (2017). *JMLR, 18*(17), 1–5. | [jmlr.org/papers/v18/16-365](http://jmlr.org/papers/v18/16-365.html) | Librería imbalanced-learn |


