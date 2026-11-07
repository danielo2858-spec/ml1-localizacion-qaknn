"""
Despliegue básico — Localización interior por RSSI WiFi (reconstrucción de QA-kNN).
Proyecto Integrador ML1 · Sebastián Ramos · Daniel Cipagauta

Ejecutar desde la carpeta del repositorio:
    streamlit run app.py

La app carga el Pipeline guardado por el notebook (modelo/modelo_final.joblib), resume las lecturas con la
misma función del entrenamiento (qaknn.simulacion.extraer_caracteristicas) y devuelve la zona estimada.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

from qaknn.simulacion import (ANCHO, APS, FONDO, N_APS, Y_CORREDOR, ZONAS, ZONA_IDX, construir_edificio,
                              extraer_caracteristicas, simular_visita)

RAIZ = Path(__file__).parent
TINTA, TINTA2, GRIS, AZUL, VERDE, ROJO = "#0b0b0b", "#52514e", "#898781", "#2a78d6", "#1baf7a", "#e34948"
K_VECINOS = 5          # vecinos que se muestran como evidencia y para el Método II
MIN_LECTURAS = 5       # con menos lecturas por AP el resumen en cuartiles deja de ser confiable

st.set_page_config(page_title="Localización interior QA-kNN", page_icon="📶", layout="wide")


@st.cache_resource
def cargar():
    modelo = joblib.load(RAIZ / "modelo" / "modelo_final.joblib")
    meta = json.loads((RAIZ / "modelo" / "metadata.json").read_text(encoding="utf-8"))
    return modelo, meta, construir_edificio(meta["semilla"])


modelo, META, EDIF = cargar()
FEATS = META["variables"]
CENTROS = np.array(META["centros"])
ETIQ = np.array(META["etiquetas_entrenamiento"])
TIPO = {z[0]: z[5] for z in ZONAS}


# ---------------------------------------------------------------- predicción y controles de coherencia
def predecir(features):
    """features: dict con las 24 variables. Devuelve la zona, los vecinos y la posición del Método II."""
    X = pd.DataFrame([features])[FEATS]
    zona = modelo.predict(X)[0]
    Z = modelo[:-1].transform(X)
    dist, idx = modelo.named_steps["modelo"].kneighbors(Z, n_neighbors=K_VECINOS)
    vecinos = ETIQ[idx[0]]
    centroide = CENTROS[[ZONA_IDX[z] for z in vecinos]].mean(axis=0)
    return zona, vecinos, dist[0], centroide


def revisar(features, n_lecturas=None):
    """Advertencias sobre entradas que el modelo no vio en el entrenamiento."""
    avisos = []
    faltan = [f"AP{a}" for a in range(1, N_APS + 1) if np.isnan(features[f"AP{a}_Q2"])]
    if faltan:
        lo, hi = META["acierto_si_cae_un_ap"]
        efecto = (f"con un AP caído, el acierto esperado baja a {lo:.0%}–{hi:.0%}" if len(faltan) == 1
                  else "con varios APs caídos, el acierto baja todavía más")
        avisos.append(f"Sin señal suficiente de {', '.join(faltan)}: se imputa con la mediana del entrenamiento; {efecto}.")
    if len(faltan) >= 3:
        avisos.append("Faltan 3 o más APs: la predicción es poco confiable.")
    fuera = [f for f in FEATS if not np.isnan(features[f])
             and not (META["rango_entrenamiento"][f][0] - 3 <= features[f] <= META["rango_entrenamiento"][f][1] + 3)]
    if fuera:
        avisos.append(f"Valores fuera del rango de entrenamiento en {', '.join(sorted({f[:3] for f in fuera}))}: "
                      "¿las lecturas son de este edificio y en dBm?")
    desorden = [f"AP{a}" for a in range(1, N_APS + 1)
                if not np.isnan(features[f"AP{a}_Q2"])
                and not (features[f"AP{a}_Q1"] <= features[f"AP{a}_Q2"] <= features[f"AP{a}_Q3"])]
    if desorden:
        avisos.append(f"En {', '.join(desorden)} no se cumple Q1 ≤ Q2 ≤ Q3.")
    if n_lecturas is not None and n_lecturas < MIN_LECTURAS:
        avisos.append(f"Solo {n_lecturas} lecturas por AP: se recomiendan al menos {MIN_LECTURAS}.")
    return avisos


def plano(zona_pred, centroide, real=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for zid, x0, x1, y0, y1, tipo in ZONAS:
        fc = {"Aula": "#eef4fa", "Corredor": "#f4f3ee", "Oficina": "#eaf6f1"}[tipo]
        if zid == zona_pred:
            fc = "#cfe2f8"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor="white", lw=1.5))
        ax.text(x0 + 0.3, y0 + 0.35, zid, fontsize=8, color=AZUL if zid == zona_pred else TINTA2,
                weight="bold" if zid == zona_pred else "normal")
    zid, x0, x1, y0, y1, _ = ZONAS[ZONA_IDX[zona_pred]]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=AZUL, lw=3, zorder=4))
    for yw in Y_CORREDOR:
        ax.plot([0, ANCHO], [yw, yw], color=TINTA2, lw=2.2)
    for xw in np.arange(5, ANCHO, 5):
        ax.plot([xw, xw], [0, 8], color=GRIS, lw=1); ax.plot([xw, xw], [12, 20], color=GRIS, lw=1)
    ax.add_patch(Rectangle((0, 0), ANCHO, FONDO, fill=False, edgecolor=TINTA, lw=2.2))
    ax.scatter(APS[:, 0], APS[:, 1], marker="^", s=60, color=ROJO, edgecolor="white", zorder=5, label="Punto de acceso")
    for i, (x, y) in enumerate(APS):
        ax.annotate(f"AP{i+1}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=ROJO, weight="bold")
    ax.scatter(*centroide, marker="X", s=140, color=VERDE, edgecolor="white", zorder=6, label="Posición aproximada (Método II)")
    if real is not None:
        ax.scatter(*real, marker="o", s=90, color=TINTA, edgecolor="white", zorder=6, label="Posición real (simulación)")
    ax.set_xlim(-0.3, ANCHO + 0.3); ax.set_ylim(-0.3, FONDO + 0.3); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def mostrar_resultado(features, n_lecturas=None, zona_real=None, pos_real=None):
    for aviso in revisar(features, n_lecturas):
        st.warning(aviso)
    zona, vecinos, dist, centroide = predecir(features)
    c1, c2, c3 = st.columns(3)
    c1.metric("Zona estimada", f"{zona} · {TIPO[zona]}")
    votos = int((vecinos == zona).sum())
    c2.metric(f"Vecinos de la misma zona (de {K_VECINOS})", f"{votos}/{K_VECINOS}",
              help="Cuántas de las visitas de entrenamiento más parecidas son de la zona estimada. "
                   "Pocas coincidencias = la visita está cerca de una frontera.")
    c3.metric("Posición aproximada (Método II)", f"x = {centroide[0]:.1f} m · y = {centroide[1]:.1f} m")
    if zona_real is not None:
        if zona == zona_real:
            st.success(f"Acierto: la visita se simuló en {zona_real}.")
        else:
            d = np.linalg.norm(CENTROS[ZONA_IDX[zona]] - CENTROS[ZONA_IDX[zona_real]])
            st.error(f"Error: la visita se simuló en {zona_real} (centros a {d:.1f} m).")
    izq, der = st.columns([1.35, 1])
    izq.pyplot(plano(zona, centroide, pos_real), clear_figure=True)
    with der:
        st.markdown("**Visitas de entrenamiento más parecidas**")
        st.dataframe(pd.DataFrame({"Zona": vecinos, "Tipo": [TIPO[z] for z in vecinos], "Distancia": dist.round(2)}),
                     hide_index=True, width="stretch")
        st.markdown("**Cuartiles calculados (dBm)**")
        tabla = pd.DataFrame([[features[f"AP{a}_{q}"] for q in ("Q1", "Q2", "Q3")] for a in range(1, N_APS + 1)],
                             index=[f"AP{a}" for a in range(1, N_APS + 1)], columns=["Q1", "Q2", "Q3"])
        st.dataframe(tabla.round(1), width="stretch")


# ---------------------------------------------------------------- interfaz
with st.sidebar:
    st.header("📶 QA-kNN")
    st.markdown("Estima la **zona** de un piso universitario (18 zonas, 8 APs) a partir del RSSI WiFi, "
                "como en Ferreira et al. (2020).")
    st.markdown(f"**Modelo:** {META['modelo']} · k = 1 · Manhattan  \n"
                f"**Prueba (días nuevos):** {META['metricas_prueba']['accuracy']:.1%} de acierto · "
                f"error medio {META['metricas_prueba']['error_medio_m']:.2f} m  \n"
                f"**Entrenado con:** {META['entrenado_con']}")
    st.info("Datos de entrenamiento **simulados** (calibrados con mediciones reales). "
            "Para usarlo en un edificio real hay que recolectar su propia huella.")
    st.caption("Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia  \n"
               "Sebastián Ramos · Daniel Cipagauta  \nUso de IA declarado: apoyo para código y redacción.")

st.title("Localización interior por WiFi")
st.markdown("Cada visita reúne varias lecturas de RSSI (dBm) de los 8 puntos de acceso. La app las resume en sus "
            "cuartiles **Q1, Q2 y Q3**, aplica el **Pipeline** entrenado (imputación → escalado → kNN) y devuelve la zona.")

tab_sim, tab_csv, tab_man = st.tabs(["🎲 Simular una visita", "📄 Cargar lecturas (CSV)", "✍️ Ingresar cuartiles"])

with tab_sim:
    st.markdown("Genera una visita con el mismo simulador del entrenamiento, en un punto al azar de la zona elegida.")
    a, b, c = st.columns([1, 1, 1.4])
    zona_sel = a.selectbox("Zona donde está el dispositivo", [z[0] for z in ZONAS], index=6,
                           format_func=lambda z: f"{z} · {TIPO[z]}")
    m_sel = b.slider("Lecturas por AP", MIN_LECTURAS, 40, 10, help="Con 5 lecturas el acierto se mantiene cerca del 97 %.")
    caidos = c.multiselect("Simular APs apagados", [f"AP{i}" for i in range(1, N_APS + 1)], placeholder="Ninguno")
    if "semilla" not in st.session_state:
        st.session_state.semilla = 0
    if st.button("Nueva visita", type="primary"):
        st.session_state.semilla += 1
    rng = np.random.default_rng(10_000 + st.session_state.semilla)
    A, p = simular_visita(EDIF, zona_sel, rng, m=m_sel, aps_caidos=[int(x[2:]) - 1 for x in caidos])
    mostrar_resultado(extraer_caracteristicas(A), m_sel, zona_sel, p)
    with st.expander("Ver las lecturas crudas de esta visita"):
        st.dataframe(pd.DataFrame(A, columns=[f"AP{i}" for i in range(1, N_APS + 1)]), width="stretch")

with tab_csv:
    st.markdown("Un archivo con una columna por AP (`AP1` … `AP8`) y una fila por lectura, en dBm. "
                "Las celdas vacías son lecturas no detectadas.")
    ejemplo, _ = simular_visita(EDIF, "Z15", np.random.default_rng(7), m=20)
    buf = io.StringIO(); pd.DataFrame(ejemplo, columns=[f"AP{i}" for i in range(1, N_APS + 1)]).to_csv(buf, index=False)
    st.download_button("Descargar un CSV de ejemplo (zona Z15)", buf.getvalue(), "visita_ejemplo.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        lect = pd.read_csv(archivo)
        lect.columns = [c.strip().upper() for c in lect.columns]
        esperadas = [f"AP{i}" for i in range(1, N_APS + 1)]
        if not set(esperadas) <= set(lect.columns):
            st.error(f"El archivo debe tener las columnas {', '.join(esperadas)}. Tiene: {', '.join(lect.columns)}.")
        else:
            A = lect[esperadas].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            st.caption(f"{len(A)} lecturas leídas.")
            mostrar_resultado(extraer_caracteristicas(A), len(A))

with tab_man:
    st.markdown("Para quien ya tiene los cuartiles por AP (por ejemplo, de otra herramienta). Deje vacío un AP sin señal.")
    base = extraer_caracteristicas(simular_visita(EDIF, "Z07", np.random.default_rng(3), m=40)[0])
    inicial = pd.DataFrame([[round(base[f"AP{a}_{q}"], 1) for q in ("Q1", "Q2", "Q3")] for a in range(1, N_APS + 1)],
                           index=[f"AP{a}" for a in range(1, N_APS + 1)], columns=["Q1", "Q2", "Q3"])
    editado = st.data_editor(inicial, width="stretch", num_rows="fixed")
    feats = {f"AP{a}_{q}": float(editado.loc[f"AP{a}", q]) if pd.notna(editado.loc[f"AP{a}", q]) else np.nan
             for a in range(1, N_APS + 1) for q in ("Q1", "Q2", "Q3")}
    mostrar_resultado(feats)

with st.expander("¿Cómo funciona y qué tan confiable es?"):
    st.markdown(f"""
1. **Lecturas → cuartiles.** Por cada AP se calculan Q1, Q2 y Q3 (Ec. 1 del artículo). Un AP detectado en menos del 25 % de las lecturas se marca como faltante.
2. **Pipeline guardado.** Imputación por mediana → escalado → kNN (k = 1, Manhattan), ajustados con {META['entrenado_con']}.
3. **Resultado.** La zona del vecino más parecido (Método I) y, como apoyo, el centroide de los {K_VECINOS} vecinos más cercanos (Método II).

**Confiabilidad:** {META['metricas_prueba']['accuracy']:.1%} de acierto en días que el modelo no vio. Cuando falla, casi siempre confunde la zona con una **contigua** (~5–6 m). Si un AP se apaga, el acierto baja a {META['acierto_si_cae_un_ap'][0]:.0%}–{META['acierto_si_cae_un_ap'][1]:.0%}.

**Límite:** el modelo conoce solo el edificio simulado. En un edificio real hay que medir su huella y reentrenar con el notebook.
""")
