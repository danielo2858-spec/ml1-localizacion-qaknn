# Dataset MNAV

Escaneos WiFi tomados en las 16 salas del Museo Nacional de Artes Visuales (MNAV) de Montevideo, Uruguay.

- **Fuente:** repositorio [ffedee7/posifi_mnav](https://github.com/ffedee7/posifi_mnav), archivo `data_analysis/anonym.csv`.
- **Licencia:** MIT (copia en `LICENSE`).
- **Contenido:** 10 469 filas. Cada fila es un escaneo; `location` es la sala (`location_1` … `location_16`) y las 188
  columnas restantes son el RSSI en dBm de APs con nombre anonimizado. Un 0 significa que el AP no se detectó.
- **Sin hora ni coordenadas:** el dataset no indica cuándo ni en qué punto exacto de la sala se tomó cada escaneo.

El notebook `Entrega_Final_Notebook.ipynb` lee este archivo y, si no está, lo descarga de la fuente.
