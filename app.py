import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta, timezone
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

st.set_page_config(page_title="Visor Satelital de Panamá", page_icon="🛰️", layout="wide")
UA = {"User-Agent": "VisorSatelitalPanama/1.0"}
LUGARES = {
    "Ciudad de Panamá": (8.98, -79.52), "Colón": (9.36, -79.90), "David (Chiriquí)": (8.43, -82.43),
    "Santiago (Veraguas)": (8.10, -80.97), "Yaviza (Darién)": (8.15, -77.69),
    "Bocas del Toro": (9.34, -82.25), "Otro (escribir coordenadas)": None,
}


# ---------- utilidades ----------
@st.cache_data(ttl=900, show_spinner=False)
def jget(url, params=None):
    r = requests.get(url, params=params, headers=UA, timeout=40)
    r.raise_for_status()
    return r.json()


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


# ---------- páginas ----------
def inicio(lat, lon):
    st.title("🛰️ Visor Satelital de Panamá")
    st.write("Elige una sección en el menú de la izquierda. Todo usa **datos públicos y gratuitos, sin claves**.")
    st.table(pd.DataFrame([
        ["🖼️ Imágenes Sentinel-2", "ESA Copernicus vía AWS Open Data", "Color real y vegetación (NDVI), 10 m"],
        ["⛅ Clima y pronóstico", "Open-Meteo", "Temperatura, lluvia, viento, 7 días"],
        ["📈 Histórico lluvia/temperatura", "NASA POWER", "Datos satelitales diarios, hasta 1 año"],
        ["🌀 Nubes en vivo (GOES)", "NOAA GOES-East", "Imagen del Caribe cada ~10 min"],
        ["🌫️ Calidad del aire", "Open-Meteo / Copernicus CAMS", "PM2.5, PM10, índice AQI"],
        ["🛰️ Satélites en órbita", "CelesTrak (NORAD)", "Posición actual de satélites"],
        ["🌋 Sismos", "USGS", "Terremotos de los últimos 30 días"],
    ], columns=["Sección", "Fuente", "Qué muestra"]))
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=7)


def imagenes(lat, lon):
    st.header("🖼️ Imágenes Sentinel-2")
    bbox = (lon - 0.15, lat - 0.12, lon + 0.15, lat + 0.12)
    c = st.columns(3)
    desde = c[0].date_input("Desde", date.today() - timedelta(days=45))
    hasta = c[1].date_input("Hasta", date.today())
    nubes = c[2].slider("Nubosidad máxima (%)", 5, 100, 40)

    @st.cache_data(ttl=3600, show_spinner="Buscando escenas...")
    def buscar(bbox, d, h, n):
        res = Client.open("https://earth-search.aws.element84.com/v1").search(
            collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{d}/{h}",
            query={"eo:cloud_cover": {"lt": n}}, max_items=20)
        return sorted([i.to_dict() for i in res.item_collection()],
                      key=lambda i: i["properties"]["datetime"], reverse=True)

    @st.cache_data(ttl=3600, show_spinner="Descargando imagen...")
    def leer(href, bbox, bandas, ancho=800):
        with rasterio.open(href) as src:
            win = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), transform=src.transform)
            alto = max(1, int(ancho * win.height / win.width))
            return src.read(list(bandas), window=win, out_shape=(len(bandas), alto, ancho))

    items = buscar(bbox, str(desde), str(hasta), nubes)
    if not items:
        return st.warning("No hay escenas con esos filtros. Sube la nubosidad o amplía las fechas.")
    et = [f'{i["properties"]["datetime"][:10]} · ☁ {i["properties"]["eo:cloud_cover"]:.0f}%' for i in items]
    k = st.selectbox("Escena", range(len(items)), format_func=lambda x: et[x])
    a = items[k]["assets"]
    rgb = leer(a["visual"]["href"], bbox, (1, 2, 3)).transpose(1, 2, 0)
    rojo = (leer(a["red"]["href"], bbox, (1,))[0].astype("float32") - 1000) / 10000
    nir = (leer(a["nir"]["href"], bbox, (1,))[0].astype("float32") - 1000) / 10000
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - rojo) / (nir + rojo)
    c1, c2 = st.columns(2)
    c1.subheader("Color real")
    c1.image(rgb, use_container_width=True)
    c2.subheader("Vegetación (NDVI)")
    fig, ax = plt.subplots()
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7, label="NDVI")
    c2.pyplot(fig)
    st.metric("NDVI medio", f"{np.nanmean(ndvi):.2f}")
    st.caption("0 = suelo o ciudad · 0.5 = cultivos o pasto · 0.8+ = selva densa")


def clima(lat, lon):
    st.header("⛅ Clima y pronóstico")
    d = jget("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 7,
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"})
    cur = d["current"]
    m = st.columns(5)
    m[0].metric("Temperatura", f'{cur["temperature_2m"]} °C')
    m[1].metric("Humedad", f'{cur["relative_humidity_2m"]} %')
    m[2].metric("Nubes", f'{cur["cloud_cover"]} %')
    m[3].metric("Viento", f'{cur["wind_speed_10m"]} km/h')
    m[4].metric("Lluvia ahora", f'{cur["precipitation"]} mm')
    df = pd.DataFrame(d["daily"]).set_index("time")
    st.subheader("Temperatura (°C)")
    st.line_chart(df[["temperature_2m_max", "temperature_2m_min"]])
    st.subheader("Lluvia prevista (mm)")
    st.bar_chart(df["precipitation_sum"])
    st.dataframe(df.rename(columns={"temperature_2m_max": "Máx °C", "temperature_2m_min": "Mín °C",
                                    "precipitation_sum": "Lluvia mm", "precipitation_probability_max": "Prob. lluvia %"}))


def historico(lat, lon):
    st.header("📈 Histórico satelital (NASA POWER)")
    dias = st.select_slider("Período", [30, 90, 180, 365], value=180)
    fin = date.today() - timedelta(days=3)  # POWER publica con unos días de retraso
    ini = fin - timedelta(days=dias)
    d = jget("https://power.larc.nasa.gov/api/temporal/daily/point", {
        "parameters": "PRECTOTCORR,T2M,ALLSKY_SFC_SW_DWN", "community": "AG", "latitude": lat, "longitude": lon,
        "start": ini.strftime("%Y%m%d"), "end": fin.strftime("%Y%m%d"), "format": "JSON"})
    df = pd.DataFrame(d["properties"]["parameter"]).replace(-999, np.nan)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    m = st.columns(3)
    m[0].metric("Lluvia total", f'{df["PRECTOTCORR"].sum():.0f} mm')
    m[1].metric("Temperatura media", f'{df["T2M"].mean():.1f} °C')
    m[2].metric("Radiación solar media", f'{df["ALLSKY_SFC_SW_DWN"].mean():.1f} kWh/m²/día')
    st.subheader("Lluvia diaria (mm)")
    st.bar_chart(df["PRECTOTCORR"])
    st.subheader("Temperatura media (°C)")
    st.line_chart(df["T2M"])
    st.subheader("Lluvia por mes (mm)")
    st.bar_chart(df["PRECTOTCORR"].resample("MS").sum())


def goes():
    st.header("🌀 Nubes en vivo · GOES-East (Caribe)")
    prod = {"Color real": "GEOCOLOR", "Infrarrojo (nubes altas = más brillantes)": "13", "Vapor de agua": "09"}
    nombre = st.radio("Producto", list(prod), horizontal=True)

    @st.cache_data(ttl=300, show_spinner="Descargando imagen de NOAA...")
    def bajar(p):
        for sat in ("GOES19", "GOES16"):  # NOAA cambió de satélite GOES-East; se prueban ambas rutas
            for tam in ("1250x750", "1000x1000", "600x600"):
                u = f"https://cdn.star.nesdis.noaa.gov/{sat}/ABI/SECTOR/car/{p}/{tam}.jpg"
                try:
                    r = requests.get(u, headers=UA, timeout=30)
                    if r.ok and r.headers.get("content-type", "").startswith("image"):
                        return r.content, u
                except requests.RequestException:
                    pass
        return None, None

    img, url = bajar(prod[nombre])
    if img is None:
        return st.error("No se pudo descargar la imagen de NOAA ahora mismo. Intenta en unos minutos.")
    st.image(img, use_container_width=True)
    st.caption(f"Fuente: NOAA/NESDIS/STAR · {url} · se actualiza cada ~10 minutos")


def aire(lat, lon):
    st.header("🌫️ Calidad del aire")
    d = jget("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 2, "forecast_days": 2,
        "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide", "hourly": "pm2_5,pm10"})
    c = d["current"]
    aqi = c["us_aqi"]
    nivel = "Buena 🟢" if aqi <= 50 else "Moderada 🟡" if aqi <= 100 else "Dañina para sensibles 🟠" if aqi <= 150 else "Dañina 🔴"
    m = st.columns(4)
    m[0].metric("Índice AQI (EE. UU.)", aqi, nivel)
    m[1].metric("PM2.5", f'{c["pm2_5"]} µg/m³')
    m[2].metric("PM10", f'{c["pm10"]} µg/m³')
    m[3].metric("Ozono", f'{c["ozone"]} µg/m³')
    df = pd.DataFrame(d["hourly"]).set_index("time")
    st.line_chart(df)
    st.caption("Modelo Copernicus CAMS (usa datos satelitales). Es una estimación, no una estación local.")


def orbitas(lat, lon):
    st.header("🛰️ Satélites en órbita ahora")
    from skyfield.api import EarthSatellite, load, wgs84
    grupos = {"Estaciones espaciales": "stations", "Meteorológicos": "weather",
              "Observación de la Tierra": "resource", "GPS": "gps-ops"}
    g = st.radio("Grupo", list(grupos), horizontal=True)

    @st.cache_data(ttl=7200, show_spinner="Descargando órbitas de CelesTrak...")
    def tle(grupo):
        r = requests.get("https://celestrak.org/NORAD/elements/gp.php",
                         params={"GROUP": grupo, "FORMAT": "tle"}, headers=UA, timeout=40)
        r.raise_for_status()
        return r.text.strip().splitlines()

    ls = tle(grupos[g])
    ts = load.timescale()
    t = ts.now()
    filas = []
    for i in range(0, len(ls) - 2, 3):
        try:
            p = wgs84.subpoint(EarthSatellite(ls[i + 1], ls[i + 2], ls[i].strip(), ts).at(t))
            filas.append({"Satélite": ls[i].strip(), "lat": p.latitude.degrees, "lon": p.longitude.degrees,
                          "Altura km": round(p.elevation.km)})
        except Exception:
            continue
    df = pd.DataFrame(filas)
    if df.empty:
        return st.warning("No se pudieron calcular posiciones.")
    df["Distancia a tu zona km"] = haversine(lat, lon, df["lat"], df["lon"]).round()
    st.map(df, zoom=1)
    cerca = df[df["Distancia a tu zona km"] < 2000]
    st.metric("Satélites a menos de 2000 km de tu zona ahora", len(cerca))
    st.dataframe(df.sort_values("Distancia a tu zona km").head(15), hide_index=True)
    st.caption(f"{len(df)} satélites · posiciones calculadas para {datetime.now(timezone.utc):%H:%M} UTC")


def sismos(lat, lon):
    st.header("🌋 Sismos (últimos 30 días, 500 km)")
    mag = st.slider("Magnitud mínima", 2.0, 6.0, 3.0, 0.5)
    d = jget("https://earthquake.usgs.gov/fdsnws/event/1/query", {
        "format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": 500, "minmagnitude": mag,
        "starttime": (date.today() - timedelta(days=30)).isoformat(), "orderby": "time", "limit": 200})
    filas = [{"Fecha": datetime.fromtimestamp(f["properties"]["time"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
              "Magnitud": f["properties"]["mag"], "Lugar": f["properties"]["place"],
              "Prof. km": f["geometry"]["coordinates"][2], "lat": f["geometry"]["coordinates"][1],
              "lon": f["geometry"]["coordinates"][0]} for f in d["features"]]
    if not filas:
        return st.info("No hubo sismos con esa magnitud en el período.")
    df = pd.DataFrame(filas)
    st.metric("Sismos encontrados", len(df))
    st.map(df, zoom=5)
    st.dataframe(df[["Fecha", "Magnitud", "Lugar", "Prof. km"]], hide_index=True)


# ---------- menú ----------
st.sidebar.title("🛰️ Menú")
seccion = st.sidebar.radio("Sección", ["🏠 Inicio", "🖼️ Imágenes Sentinel-2", "⛅ Clima y pronóstico",
                                       "📈 Histórico lluvia/temperatura", "🌀 Nubes en vivo (GOES)",
                                       "🌫️ Calidad del aire", "🛰️ Satélites en órbita", "🌋 Sismos"])
lugar = st.sidebar.selectbox("📍 Ubicación", list(LUGARES))
if LUGARES[lugar]:
    lat, lon = LUGARES[lugar]
else:
    lat = st.sidebar.number_input("Latitud", -90.0, 90.0, 8.98, 0.01)
    lon = st.sidebar.number_input("Longitud", -180.0, 180.0, -79.52, 0.01)
st.sidebar.caption(f"{lat:.2f}, {lon:.2f}")

paginas = {"🏠 Inicio": lambda: inicio(lat, lon), "🖼️ Imágenes Sentinel-2": lambda: imagenes(lat, lon),
           "⛅ Clima y pronóstico": lambda: clima(lat, lon), "📈 Histórico lluvia/temperatura": lambda: historico(lat, lon),
           "🌀 Nubes en vivo (GOES)": goes, "🌫️ Calidad del aire": lambda: aire(lat, lon),
           "🛰️ Satélites en órbita": lambda: orbitas(lat, lon), "🌋 Sismos": lambda: sismos(lat, lon)}
try:
    paginas[seccion]()
except requests.RequestException as e:
    st.error(f"No se pudo conectar con la fuente de datos: {e}")
except Exception as e:
    st.error(f"Ocurrió un error: {e}")
