"""
Piloto real de calibración de RSSI WiFi (Windows) — Proyecto Integrador ML1.

Registra lecturas RSSI (dBm) de los puntos de acceso visibles usando la API nativa
de Windows (wlanapi.dll: WlanScan + WlanGetNetworkBssList). No guarda nombres de red:
cada BSSID se anonimiza con un hash corto.

Uso (en el campus, de pie en cada punto de medición, sin moverse):
    python piloto_rssi_windows.py --punto P1_pasillo --lecturas 40 --salida datos_piloto/piloto_campus.csv
    python piloto_rssi_windows.py --punto P2_aula --lecturas 40 --salida datos_piloto/piloto_campus.csv
    ...
Cada ejecución agrega filas al CSV: timestamp, punto, ap_hash, banda, rssi_dbm.
"""
import argparse
import ctypes
import csv
import hashlib
import os
import time
from ctypes import wintypes

wlanapi = ctypes.WinDLL("wlanapi.dll")


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


class WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [("InterfaceGuid", GUID),
                ("strInterfaceDescription", ctypes.c_wchar * 256),
                ("isState", ctypes.c_uint)]


class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
    _fields_ = [("dwNumberOfItems", wintypes.DWORD), ("dwIndex", wintypes.DWORD),
                ("InterfaceInfo", WLAN_INTERFACE_INFO * 1)]


class DOT11_SSID(ctypes.Structure):
    _fields_ = [("uSSIDLength", wintypes.ULONG), ("ucSSID", ctypes.c_char * 32)]


class WLAN_RATE_SET(ctypes.Structure):
    _fields_ = [("uRateSetLength", wintypes.ULONG), ("usRateSet", wintypes.USHORT * 126)]


class WLAN_BSS_ENTRY(ctypes.Structure):
    _fields_ = [("dot11Ssid", DOT11_SSID), ("uPhyId", wintypes.ULONG),
                ("dot11Bssid", ctypes.c_ubyte * 6), ("dot11BssType", ctypes.c_uint),
                ("dot11BssPhyType", ctypes.c_uint), ("lRssi", wintypes.LONG),
                ("uLinkQuality", wintypes.ULONG), ("bInRegDomain", wintypes.BOOLEAN),
                ("usBeaconPeriod", wintypes.USHORT), ("ullTimestamp", ctypes.c_ulonglong),
                ("ullHostTimestamp", ctypes.c_ulonglong), ("usCapabilityInformation", wintypes.USHORT),
                ("ulChCenterFrequency", wintypes.ULONG), ("wlanRateSet", WLAN_RATE_SET),
                ("ulIeOffset", wintypes.ULONG), ("ulIeSize", wintypes.ULONG)]


class WLAN_BSS_LIST(ctypes.Structure):
    _fields_ = [("dwTotalSize", wintypes.DWORD), ("dwNumberOfItems", wintypes.DWORD),
                ("wlanBssEntries", WLAN_BSS_ENTRY * 1)]


def abrir_interfaz():
    handle = wintypes.HANDLE()
    version = wintypes.DWORD()
    if wlanapi.WlanOpenHandle(2, None, ctypes.byref(version), ctypes.byref(handle)) != 0:
        raise RuntimeError("No se pudo abrir el manejador WLAN")
    lista = ctypes.POINTER(WLAN_INTERFACE_INFO_LIST)()
    if wlanapi.WlanEnumInterfaces(handle, None, ctypes.byref(lista)) != 0:
        raise RuntimeError("No se pudieron enumerar las interfaces WLAN")
    if lista.contents.dwNumberOfItems == 0:
        raise RuntimeError("No hay interfaz WiFi")
    guid = lista.contents.InterfaceInfo[0].InterfaceGuid
    return handle, guid


def escanear(handle, guid, espera_s):
    wlanapi.WlanScan(handle, ctypes.byref(guid), None, None, None)
    time.sleep(espera_s)
    bss = ctypes.POINTER(WLAN_BSS_LIST)()
    if wlanapi.WlanGetNetworkBssList(handle, ctypes.byref(guid), None, 0, False, None, ctypes.byref(bss)) != 0:
        return []
    n = bss.contents.dwNumberOfItems
    entradas = ctypes.cast(ctypes.addressof(bss.contents.wlanBssEntries),
                           ctypes.POINTER(WLAN_BSS_ENTRY * n)).contents
    filas = []
    for e in entradas:
        mac = ":".join(f"{b:02x}" for b in e.dot11Bssid)
        ap_hash = hashlib.sha1(mac.encode()).hexdigest()[:8]
        banda = "5GHz" if e.ulChCenterFrequency > 4_000_000 else "2.4GHz"
        filas.append((ap_hash, banda, int(e.lRssi)))
    wlanapi.WlanFreeMemory(bss)
    return filas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--punto", required=True, help="Etiqueta del punto de medición, p. ej. P1_pasillo")
    p.add_argument("--lecturas", type=int, default=40, help="Número de escaneos en el punto")
    p.add_argument("--espera", type=float, default=4.0, help="Segundos entre escaneo y lectura")
    p.add_argument("--salida", default="datos_piloto/piloto_campus.csv")
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    nuevo = not os.path.exists(a.salida)
    handle, guid = abrir_interfaz()
    with open(a.salida, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["timestamp", "punto", "ap_hash", "banda", "rssi_dbm"])
        for i in range(a.lecturas):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            filas = escanear(handle, guid, a.espera)
            for ap_hash, banda, rssi in filas:
                w.writerow([ts, a.punto, ap_hash, banda, rssi])
            print(f"[{i + 1}/{a.lecturas}] {a.punto}: {len(filas)} APs visibles")
    wlanapi.WlanCloseHandle(handle, None)


if __name__ == "__main__":
    main()
