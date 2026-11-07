"""
Simulación fundamentada de RSSI WiFi para un piso de edificio universitario colombiano.
Proyecto Integrador ML1 — Entrega final (Sebastián Ramos · Daniel Cipagauta).

Modelo generador (una lectura del AP a en la posición p, día t, instante j):

    RSSI = P0_a − 10·n·log10(d/d0) − Σ_paredes L_w  +  S_a(p)  +  D_{t,a}  +  e_j  −  F_j

    P0_a      potencia a 1 m de cada AP (hardware distinto)          ~ N(−41, 2²) dBm
    n         exponente de pérdida de trayecto (con paredes explícitas) = 2.2
    L_w       atenuación por pared atravesada (COST 231 multipared): 6 dB muro de corredor, 4 dB tabique
    S_a(p)    sombra espacial: campo gaussiano con correlación exponencial (σ = 5 dB, 3 m) — fijo en el sitio
    D_{t,a}   deriva diaria por AP (ocupación, puertas, humedad)       ~ N(0, 1²) dB
    e_j       fluctuación temporal AR(1) (φ = 0.3), σ_t = sigma_base + sigma_pared·paredes dB
    F_j       desvanecimiento profundo ocasional: prob. fade_base + fade_pared·paredes, magnitud ~ Exp(fade_media)

Dos escenarios para e_j y F_j (ESCENARIOS):
    "calibrado"  escenario principal de la entrega final: fluctuación ~1 dB en todo el piso y caídas raras,
                 sin dependencia de las paredes; reproduce la dispersión y la asimetría de los datos reales.
    "paredes"    escenario de la Segunda entrega: la fluctuación y las caídas crecen con las paredes
                 (supuesto de la literatura). Deja la dispersión en ~2× la real; se usa como sensibilidad.

Las lecturas se redondean a dBm enteros. Cada lectura se detecta con probabilidad logística
p = 1 / (1 + exp(−(RSSI + 91) / 2)) (escaneos que no captan el beacon de APs débiles) y nunca por
debajo de la sensibilidad (−95 dBm). σ_S, la fluctuación, φ y la deriva están calibrados con las
mediciones reales publicadas por los autores del artículo (Wi-FiPowerInOfficeDataset, Manaus, Brasil);
la curva de detección, con el piloto propio.
"""
import numpy as np
import pandas as pd

# --- Geometría del piso (30 m × 20 m): aulas abajo, corredor central, aulas/oficinas arriba ---
ANCHO, FONDO = 30.0, 20.0
COLS = 6
Y_CORREDOR = (8.0, 12.0)
FILAS = [(0.0, 8.0, "Aula"), (8.0, 12.0, "Corredor"), (12.0, 20.0, "Oficina")]

ZONAS = []  # (id, x0, x1, y0, y1, tipo)
for fila, (y0, y1, tipo) in enumerate(FILAS):
    for c in range(COLS):
        zid = fila * COLS + c + 1
        ZONAS.append((f"Z{zid:02d}", c * 5.0, (c + 1) * 5.0, y0, y1, tipo))
ZONA_IDX = {z[0]: i for i, z in enumerate(ZONAS)}
CENTROS = np.array([[(z[1] + z[2]) / 2, (z[3] + z[4]) / 2] for z in ZONAS])

# 8 APs: 4 en el corredor (instalación típica de red universitaria) y 4 dentro de aulas/oficinas,
# en posiciones no simétricas para que el piso no tenga zonas "espejo".
APS = np.array([
    [3.0, 10.0], [12.5, 10.0], [19.0, 10.0], [27.5, 10.0],   # corredor
    [7.5, 3.0], [23.0, 5.5],                                 # aulas
    [2.5, 16.5], [16.5, 17.5],                               # oficinas
])
N_APS = len(APS)

P0_MEDIA, P0_SD = -41.0, 2.0
N_PL = 2.2
L_CORREDOR, L_TABIQUE = 6.0, 4.0
SIGMA_SOMBRA, DIST_CORR = 5.0, 3.0
SIGMA_DIA = 1.0
PHI_AR = 0.3
SENSIBILIDAD = -95.0
DET_CENTRO, DET_ESCALA = -91.0, 2.0
P_FALLA_AP = 0.03
FRACCION_MIN_DETECCIONES = 0.25   # 10 de 40 lecturas

ESCENARIOS = {
    "calibrado": dict(sigma_base=0.9, sigma_pared=0.0, fade_base=0.01, fade_pared=0.0, fade_media=3.0),
    "paredes": dict(sigma_base=0.8, sigma_pared=0.35, fade_base=0.02, fade_pared=0.03, fade_media=4.0),
}


def paredes_atravesadas(p, ap):
    """Cuenta muros de corredor (y=8, y=12) y tabiques (x=5,10,...,25 fuera del corredor)
    que cruza el segmento p→ap. Devuelve (n_muros, n_tabiques)."""
    (x1, y1), (x2, y2) = p, ap
    muros = sum(1 for yw in Y_CORREDOR if (y1 - yw) * (y2 - yw) < 0)
    tabiques = 0
    for xw in np.arange(5.0, ANCHO, 5.0):
        if (x1 - xw) * (x2 - xw) < 0:
            t = (xw - x1) / (x2 - x1)
            yc = y1 + t * (y2 - y1)
            if not (Y_CORREDOR[0] < yc < Y_CORREDOR[1]):
                tabiques += 1
    return muros, tabiques


def campo_sombra(rng, res=0.5):
    """Campo gaussiano 2D con covarianza exponencial (Gudmundson) para cada AP."""
    xs = np.arange(0, ANCHO + res / 2, res)
    ys = np.arange(0, FONDO + res / 2, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    L = np.linalg.cholesky(SIGMA_SOMBRA ** 2 * np.exp(-d / DIST_CORR) + 1e-6 * np.eye(len(pts)))
    campos = np.stack([(L @ rng.standard_normal(len(pts))).reshape(gy.shape) for _ in range(N_APS)])
    return xs, ys, campos


def construir_edificio(seed=42):
    """Parámetros fijos del 'edificio': potencia de cada AP y mapa de sombra espacial."""
    rng = np.random.default_rng(seed)
    p0 = rng.normal(P0_MEDIA, P0_SD, N_APS)
    xs, ys, campos = campo_sombra(rng)
    return dict(p0=p0, xs=xs, ys=ys, campos=campos)


def nivel_medio(edif, p, a):
    """Nivel determinístico + sombra espacial (sin ruido temporal) del AP a en la posición p."""
    d = max(np.hypot(*(p - APS[a])), 1.0)
    muros, tabiques = paredes_atravesadas(p, APS[a])
    ix = int(round(p[0] / 0.5)); iy = int(round(p[1] / 0.5))
    sombra = edif["campos"][a, iy, ix]
    nivel = edif["p0"][a] - 10 * N_PL * np.log10(d) - muros * L_CORREDOR - tabiques * L_TABIQUE + sombra
    return nivel, muros + tabiques


def lecturas_visita(edif, p, deriva_dia, rng, m=40, escenario="calibrado"):
    """Matriz m × N_APS de lecturas RSSI (dBm enteros, NaN si no se detecta) en una visita."""
    esc = ESCENARIOS[escenario]
    A = np.full((m, N_APS), np.nan)
    for a in range(N_APS):
        nivel, paredes = nivel_medio(edif, p, a)
        sigma_t = esc["sigma_base"] + esc["sigma_pared"] * paredes
        e = np.empty(m)
        e[0] = rng.normal(0, sigma_t)
        innov = rng.normal(0, sigma_t * np.sqrt(1 - PHI_AR ** 2), m)
        for j in range(1, m):
            e[j] = PHI_AR * e[j - 1] + innov[j]
        p_fade = min(esc["fade_base"] + esc["fade_pared"] * paredes, 0.2)
        fades = (rng.random(m) < p_fade) * rng.exponential(esc["fade_media"], m)
        r = np.round(nivel + deriva_dia[a] + e - fades)
        detectada = rng.random(m) < 1 / (1 + np.exp(-(r - DET_CENTRO) / DET_ESCALA))
        r[(r < SENSIBILIDAD) | ~detectada] = np.nan
        A[:, a] = r
    return A


def extraer_caracteristicas(A, min_detecciones=None):
    """Resume una matriz de lecturas (m × N_APS, NaN = no detectada) en Q1, Q2, Q3 por AP, como en la
    Ec. 1 del artículo. Un AP detectado en menos del 25 % de las lecturas (10 de 40) queda como faltante.
    Es la misma función en el entrenamiento y en la aplicación de despliegue."""
    if min_detecciones is None:
        min_detecciones = max(2, int(np.ceil(FRACCION_MIN_DETECCIONES * A.shape[0])))
    fila = {}
    for a in range(A.shape[1]):
        v = A[:, a][~np.isnan(A[:, a])]
        ok = len(v) >= min_detecciones
        q = np.percentile(v, [25, 50, 75]) if ok else [np.nan] * 3
        fila[f"AP{a+1}_Q1"], fila[f"AP{a+1}_Q2"], fila[f"AP{a+1}_Q3"] = q
        fila[f"AP{a+1}_media"] = v.mean() if ok else np.nan
        fila[f"AP{a+1}_cruda"] = A[0, a]
    return fila


def generar_dataset(n_dias=14, visitas_por_dia=20, m=40, seed=42, escenario="calibrado",
                    min_detecciones=None):
    """Genera el dataset de instancias (una visita = una fila) con:
    - 24 predictores Q1, Q2, Q3 por AP (representación del artículo),
    - columnas auxiliares para comparar representaciones (media y lectura cruda única),
    - metadatos: zona (objetivo), día (grupo), posición real x, y (NO son predictores).
    `seed` fija el edificio (potencias y sombra) y la secuencia de visitas."""
    edif = construir_edificio(seed)
    rng = np.random.default_rng(seed + 1)
    filas = []
    for dia in range(1, n_dias + 1):
        deriva = rng.normal(0, SIGMA_DIA, N_APS)
        for zid, x0, x1, y0, y1, tipo in ZONAS:
            for _ in range(visitas_por_dia):
                p = np.array([rng.uniform(x0 + 0.3, x1 - 0.3), rng.uniform(y0 + 0.3, y1 - 0.3)])
                A = lecturas_visita(edif, p, deriva, rng, m, escenario)
                caidos = rng.random(N_APS) < P_FALLA_AP          # AP apagado/reiniciándose
                A[:, caidos] = np.nan
                fila = extraer_caracteristicas(A, min_detecciones)
                fila.update(zona=zid, tipo=tipo, dia=dia, x=p[0], y=p[1])
                filas.append(fila)
    # Etiquetas como objetos de Python: con pyarrow instalado, pandas 3 las guardaría como arreglos
    # Arrow, que scikit-learn no puede indexar al partir los pliegues.
    return pd.DataFrame(filas).astype({"zona": object, "tipo": object}), edif


def simular_visita(edif, zona, rng, m=40, escenario="calibrado", aps_caidos=()):
    """Una visita en una posición aleatoria de `zona` (para la demostración de la app).
    Devuelve (matriz de lecturas, posición real)."""
    zid, x0, x1, y0, y1, _ = ZONAS[ZONA_IDX[zona]]
    p = np.array([rng.uniform(x0 + 0.3, x1 - 0.3), rng.uniform(y0 + 0.3, y1 - 0.3)])
    A = lecturas_visita(edif, p, rng.normal(0, SIGMA_DIA, N_APS), rng, m, escenario)
    for a in aps_caidos:
        A[:, a] = np.nan
    return A, p


def columnas(representacion="cuartiles", aps=None):
    aps = range(1, N_APS + 1) if aps is None else aps
    if representacion == "cuartiles":
        return [f"AP{a}_{q}" for a in aps for q in ("Q1", "Q2", "Q3")]
    if representacion == "mediana":
        return [f"AP{a}_Q2" for a in aps]
    if representacion == "media":
        return [f"AP{a}_media" for a in aps]
    if representacion == "cruda":
        return [f"AP{a}_cruda" for a in aps]
    raise ValueError(representacion)