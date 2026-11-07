"""
Genera dashboard/index.html: un panel autocontenido que corre en el navegador el mismo modelo desplegado.

Exporta del modelo guardado (modelo/modelo_final.joblib) la imputación, el escalado y las 5 040 visitas de
entrenamiento ya transformadas; del simulador (qaknn) la geometría y el edificio; y de resultados_final.json las
cifras de la entrega final. Ejecutar desde la raíz del repositorio, después del notebook:

    python dashboard/construir_dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from qaknn import simulacion as sim  # noqa: E402

modelo = joblib.load(RAIZ / "modelo" / "modelo_final.joblib")
meta = json.loads((RAIZ / "modelo" / "metadata.json").read_text(encoding="utf-8"))
res = json.loads((RAIZ / "resultados_final.json").read_text(encoding="utf-8"))
df = pd.read_csv(RAIZ / "datos" / "dataset_final_calibrado.csv")
FEATS = meta["variables"]
edif = sim.construir_edificio(meta["semilla"])

# El modelo desplegado se entrenó con las 14 jornadas: sus vecinos son estas visitas transformadas.
Z = modelo[:-1].transform(df[FEATS])
zonas = [z[0] for z in sim.ZONAS]
knn = modelo.named_steps["modelo"]

datos = {
    "geometria": {
        "ancho": sim.ANCHO, "fondo": sim.FONDO, "corredor": list(sim.Y_CORREDOR),
        "zonas": [dict(id=z[0], x0=z[1], x1=z[2], y0=z[3], y1=z[4], tipo=z[5]) for z in sim.ZONAS],
        "aps": sim.APS.tolist(),
    },
    "edificio": {
        "p0": np.round(edif["p0"], 3).tolist(),
        "res": 0.5,
        "campos": np.round(edif["campos"], 2).tolist(),
        "n_pl": sim.N_PL, "l_corredor": sim.L_CORREDOR, "l_tabique": sim.L_TABIQUE,
        "sigma_dia": sim.SIGMA_DIA, "phi": sim.PHI_AR, "sensibilidad": sim.SENSIBILIDAD,
        "det_centro": sim.DET_CENTRO, "det_escala": sim.DET_ESCALA,
        "escenario": sim.ESCENARIOS["calibrado"], "fraccion_min": sim.FRACCION_MIN_DETECCIONES, "p_falla": sim.P_FALLA_AP,
    },
    "modelo": {
        "variables": FEATS,
        "imputacion": modelo.named_steps["imputar"].statistics_.round(4).tolist(),
        "media": modelo.named_steps["escalar"].mean_.round(6).tolist(),
        "escala": modelo.named_steps["escalar"].scale_.round(6).tolist(),
        "p": int(knn.p),
        "x_mil": np.round(Z * 1000).astype(int).ravel().tolist(),      # visitas × 24, en milésimas
        "y": [zonas.index(z) for z in df.zona],
        "rango": meta["rango_entrenamiento"],
        "metricas": meta["metricas_prueba"],
    },
    "resultados": {k: res[k] for k in ("comparacion", "estabilidad", "mcnemar", "barrido_nk", "representaciones",
                                       "dif_cuartiles_media", "metodos", "interpretacion", "robustez", "calibracion",
                                       "calidad", "lasso_path", "coherencia")},
}

plantilla = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
assert "/*__DATOS__*/null" in plantilla
html = plantilla.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
(Path(__file__).parent / "index.html").write_text(html, encoding="utf-8")
print(f"dashboard/index.html: {len(html) / 1e6:.2f} MB")

# Conjunto de verificación (no va en la página): visitas nuevas y la predicción del modelo de Python
if "--verificacion" in sys.argv:
    rng = np.random.default_rng(2026)
    casos = []
    for _ in range(300):
        z = zonas[rng.integers(len(zonas))]
        m = int(rng.choice([5, 10, 40]))
        caidos = list(rng.choice(8, size=rng.integers(0, 3), replace=False))
        A, _ = sim.simular_visita(edif, z, rng, m=m, aps_caidos=caidos)
        f = sim.extraer_caracteristicas(A)
        casos.append(dict(lecturas=[[None if np.isnan(v) else float(v) for v in fila] for fila in A],
                          features=[None if np.isnan(f[c]) else float(f[c]) for c in FEATS],
                          pred=str(modelo.predict(pd.DataFrame([f])[FEATS])[0])))
    salida = Path(sys.argv[sys.argv.index("--verificacion") + 1])
    salida.write_text(json.dumps(casos), encoding="utf-8")
    print("verificación:", salida, len(casos), "casos")
