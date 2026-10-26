# Localización interior mediante RSSI WiFi y kNN

**Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia**
Sebastián Ramos · Daniel Cipagauta — Prof. Diana Pulido

Reconstrucción del artículo *QA-kNN: Indoor Localization Based on Quartile Analysis and the kNN Classifier for Wireless Networks*
(Ferreira, Souza y Carvalho, 2020, *Sensors* 20(17), 4714, [doi:10.3390/s20174714](https://doi.org/10.3390/s20174714)) sobre un piso
simulado de un edificio universitario colombiano: predecir la zona (18 clases) en la que está un dispositivo a partir de los
cuartiles Q1, Q2 y Q3 del RSSI de 8 puntos de acceso.

## Estructura

| Ruta | Contenido |
|---|---|
| `Segunda_Entrega_Notebook.ipynb` | **Código de la Segunda entrega**: simulación, anclaje con datos reales, Pipeline sin fuga, comparación de modelos, interpretación |
| `Segunda_Entrega_Informe.docx` | Informe de la Segunda entrega (Fase 2: datos, modelamiento y validación) |
| `Segunda_Entrega_Presentacion.pptx` | Presentación de la sustentación (con notas del orador) |
| `Guion_Sustentacion_Segunda_Entrega.docx` | Guion de la sustentación y banco de preguntas |
| `figuras/` | Figuras del informe, generadas por el notebook |
| `resultados.json` | Todas las cifras citadas en el informe, generadas por el notebook |
| `datos/dataset_simulado_segunda_entrega.csv` | Dataset definitivo (5 040 instancias, 24 predictores + metadatos) |
| `datos_reales_articulo/` | Mediciones reales publicadas por los autores del artículo (licencia MIT) |
| `datos_piloto/` | Piloto propio de calibración (BSSID anonimizados) |
| `piloto_rssi_windows.py` | Script para medir RSSI en dBm con la API WiFi de Windows |
| `primera_entrega/` | Informe y prototipo de la Primera entrega |

## Cómo reproducir

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Segunda_Entrega_Notebook.ipynb
```

El notebook corre de principio a fin en ~1 minuto con `SEED = 42` y regenera `figuras/`, `resultados.json` y el dataset.
Si falta `datos_reales_articulo/`, lo descarga del repositorio de los autores.

Para repetir el piloto (Windows, quietos en cada punto de medición):

```bash
python piloto_rssi_windows.py --punto P1_pasillo --lecturas 40 --salida datos_piloto/piloto_campus.csv
```

El notebook incorpora automáticamente cualquier CSV nuevo de `datos_piloto/`.

## Resultados principales (Segunda entrega)

Validación cruzada agrupada por día (`GroupKFold`) sobre los días 1–10 y prueba en los días 11–14, nunca vistos:

| Modelo | Accuracy CV | Accuracy prueba [IC 95 %] | Error medio (m) |
|---|---|---|---|
| **kNN (k = 1, Manhattan)** | **0.960 ± 0.013** | **0.964 [0.954–0.973]** | **0.22** |
| QDA | 0.934 ± 0.008 | 0.942 [0.928–0.954] | 0.38 |
| Regresión logística | 0.918 ± 0.013 | 0.924 [0.910–0.937] | 0.47 |
| LDA | 0.909 ± 0.013 | 0.901 [0.885–0.915] | 0.58 |
| Naive Bayes | 0.909 ± 0.010 | 0.900 [0.883–0.915] | 0.59 |
| Línea base (mayoría) | 0.056 | 0.056 | 15.23 |

* kNN supera a QDA con diferencia significativa (McNemar exacto, p = 0.0007).
* Sobre los datos reales del artículo, el mismo flujo obtiene 100 % de acierto con 4 APs y k = 1 (su resultado de ME = 0 m).
* Resumir las lecturas aporta +7 a +12 puntos frente a una lectura cruda; Q1 y Q3 superan a la media solo con pocos APs.

## Fuentes de datos

* Ferreira, D. y Carvalho, C. (2020). *Wi-FiPowerInOfficeDataset*. [ferreirad08/DSciBox](https://github.com/ferreirad08/DSciBox), licencia MIT
  (copia en `datos_reales_articulo/LICENSE_DSciBox.txt`).
* Dataset simulado y piloto: elaboración propia.

## Uso de IA

Usamos un asistente de IA generativa (Claude, de Anthropic) como apoyo para estructurar el código, proponer verificaciones y revisar la
redacción. El equipo revisó y validó cada resultado y puede sustentar todas las decisiones técnicas.
