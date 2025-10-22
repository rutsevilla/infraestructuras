# functions.py (limpio)

from __future__ import annotations
import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import FastMarkerCluster
from folium.features import GeoJsonTooltip

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import Affine
from PIL import Image
from matplotlib import cm as mpl_cm


# ================== UTILIDADES BÁSICAS ==================

def img_to_data_uri(path: str) -> str:
    """Convierte una imagen (PNG/SVG) a data URI."""
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def load_gdf_wgs84(path: str) -> gpd.GeoDataFrame:
    """Carga cualquier capa vectorial y la devuelve en WGS84 (EPSG:4326),
    filtrando geometrías vacías y reparando inválidas si es necesario."""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    # Repara geometrías solo cuando sea necesario (buffer(0) global es caro)
    if hasattr(gdf.geometry, "is_valid"):
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    if gdf.crs is None:
        # Si conoces el CRS real, configúralo antes de convertir
        gdf = gdf.set_crs(4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


# ================== GEOJSON/SHP → GEOJSON SIMPLIFICADO ==================

def make_geojson_simplified(
    shp_path: str,
    out_geojson_path: str,
    key_col: str = "COD_UGED",
    name_candidates=("NOMBRE", "NOM_UGED", "NOM_DIST", "Distrito", "DIST_NAME"),
    tol_m: float = 150.0,
) -> str:
    """Genera un GeoJSON simplificado (tolerancia en metros) y en WGS84.
    Si ya existe y es más reciente que el SHP, lo reutiliza.
    """
    if os.path.exists(out_geojson_path):
        if os.path.getmtime(out_geojson_path) >= os.path.getmtime(shp_path):
            return out_geojson_path

    # Lectura (pyogrio si está disponible)
    try:
        gdf = gpd.read_file(shp_path, engine="pyogrio")
    except Exception:
        gdf = gpd.read_file(shp_path)

    # Repara geometrías cuando haga falta
    if hasattr(gdf.geometry, "is_valid"):
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Proyecta a métrico (Web Mercator) para simplificar por metros
    if gdf.crs is None or gdf.crs.to_epsg() != 3857:
        if gdf.crs is None:
            # Si conoces el CRS original, defínelo aquí antes
            pass
        gdf_m = gdf.to_crs(3857) if gdf.crs is not None else gdf
    else:
        gdf_m = gdf

    gdf_m["geometry"] = gdf_m.geometry.simplify(tol_m, preserve_topology=True)
    gdf_wgs = gdf_m.to_crs(4326)

    # Selección mínima de columnas
    keep_cols = [c for c in [key_col] if c in gdf_wgs.columns]
    name_col = next((c for c in name_candidates if c in gdf_wgs.columns), None)
    if name_col:
        keep_cols.append(name_col)
    keep_cols = list(dict.fromkeys(keep_cols))

    gdf_wgs = gdf_wgs[keep_cols + ["geometry"]]
    gdf_wgs.to_file(out_geojson_path, driver="GeoJSON")
    return out_geojson_path


@st.cache_data(show_spinner=False)
def load_districts_geojson_simplified(
    path: str, key_field: str, simplify_tol: float = 0.0008
) -> tuple[Dict, List[float]]:
    """Carga SHP, repara/simplifica y devuelve (geojson_dict, bounds). Simplificación en grados.
    Incluye propiedades: key_field, NOMB_UGED y NOMB_UGEC (si existen).
    """
    gdf = gpd.read_file(path).copy()
    if gdf.empty:
        raise ValueError(f"El archivo {path} está vacío.")

    # Asegura WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Filtra geometrías válidas
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    # Repara inválidas
    if hasattr(gdf.geometry, "is_valid"):
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Columnas a conservar (según disponibilidad)
    cols_wanted = [key_field, "NOMB_UGED", "NOMB_UGEC"]
    cols_present = [c for c in cols_wanted if c in gdf.columns]
    if not cols_present:
        cols_present = [key_field] if key_field in gdf.columns else []

    gdf = gdf[cols_present + ["geometry"]].copy()

    # Simplificación topológica (grados)
    gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)

    geojson = json.loads(gdf.to_json())
    bounds = gdf.total_bounds.tolist()  # [minx, miny, maxx, maxy]
    return geojson, bounds


# ================== RÁSTER → DATA URI PNG + BOUNDS ==================

@st.cache_data(show_spinner=False)
def raster_to_datauri_bounds(
    path: str,
    max_size: int = 1024,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Abre un ráster (banda 1), lo reproyecta a EPSG:4326, lo reduce y devuelve (data_uri_png, bounds, vmin, vmax)."""
    with rasterio.open(path) as src:
        # VRT reproyectado a WGS84
        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            # Tamaño destino manteniendo aspecto
            scale = min(1.0, max_size / max(vrt.width, vrt.height))
            out_w = max(1, int(vrt.width * scale))
            out_h = max(1, int(vrt.height * scale))

            # Leer banda 1 reescalada
            data = vrt.read(1, out_shape=(out_h, out_w)).astype("float32")

            # Transform reescalado
            new_transform = vrt.transform * Affine.scale(vrt.width / out_w, vrt.height / out_h)

            # Bounds (lat/lon en formato [[S,W],[N,E]])
            minx, miny = new_transform * (0, out_h)
            maxx, maxy = new_transform * (out_w, 0)
            bounds = [[miny, minx], [maxy, maxx]]

            # Normalización robusta
            if vmin is None or vmax is None:
                finite = np.isfinite(data)
                if not finite.any():
                    raise ValueError("La banda no tiene valores finitos")
                vmin = float(np.nanpercentile(data[finite], 2))
                vmax = float(np.nanpercentile(data[finite], 98))
                if vmin == vmax:
                    vmax = vmin + 1e-6

            norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            cmap_fn = mpl_cm.get_cmap(cmap)
            rgba = (cmap_fn(norm) * 255).astype(np.uint8)  # [H,W,4]

            # Transparencia donde no hay datos
            mask = ~np.isfinite(data)
            rgba[..., 3][mask] = 0

            # PNG a data URI
            im = Image.fromarray(rgba, mode="RGBA")
            buf = BytesIO()
            im.save(buf, format="PNG", optimize=True)
            data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return data_uri, bounds, float(vmin), float(vmax)


# ================== CARGA DE PUNTOS Y JSON DE ÍNDICES ==================

@st.cache_data(show_spinner=True)
def load_points(path: str) -> gpd.GeoDataFrame:
    """Carga puntos, asegura WGS84, filtra vacíos y valida tipo geométrico."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El archivo {path} está vacío o no contiene geometrías.")

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    geom_types = gdf.geom_type.unique()
    if not any(gt.startswith("Point") for gt in geom_types):
        raise TypeError(f"El archivo {path} no contiene geometrías de tipo punto. Tipos encontrados: {geom_types}")

    return gdf


@st.cache_data(show_spinner=False)
def load_indices_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_category_index_options(indices_data: Dict, cats_config: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Devuelve solo los índices que existen en el JSON para cada categoría."""
    cats: Dict[str, List[str]] = {}
    for cat, idx_list in cats_config.items():
        cats[cat] = [idx for idx in idx_list if idx in indices_data]
    return cats


def series_from_arraystyle_index(indices_data: Dict, index_key: str, code_key: str) -> pd.Series:
    """Convierte JSON 'array-style' en Serie con índice = códigos."""
    if code_key not in indices_data or index_key not in indices_data:
        return pd.Series(dtype=float)
    codes = list(map(str, indices_data[code_key]))
    vals = indices_data[index_key]
    n = min(len(codes), len(vals))
    s = pd.Series(vals[:n], index=[str(c) for c in codes[:n]], dtype=float)
    s.index.name = code_key
    s.name = index_key
    return s


# ================== ESTILOS / MAPA ==================

def add_point_layer(
    gdf_points: gpd.GeoDataFrame,
    m: folium.Map,
    layer_name: str,
    radius: int = 3,
    color: str = "#004E98",
    fill_color: str = "#07A9E0",
    fill_opacity: float = 0.9,
    tooltip_col: Optional[str] = None,
):
    """Añade puntos con CircleMarker (útil para pocos puntos)."""
    fg = folium.FeatureGroup(name=layer_name, show=True)
    for _, row in gdf_points.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        tip = None
        if tooltip_col and (tooltip_col in row) and pd.notnull(row[tooltip_col]):
            tip = str(row[tooltip_col])
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=radius,
            color=color,
            weight=1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            tooltip=tip or layer_name,
        ).add_to(fg)
    fg.add_to(m)


def add_point_cluster(gdf_points: gpd.GeoDataFrame, m: folium.Map, layer_name: str):
    """Añade puntos agrupados eficientemente (ideal para muchos puntos)."""
    coords = [(pt.y, pt.x) for pt in gdf_points.geometry if pt and not pt.is_empty]
    fg = folium.FeatureGroup(name=layer_name, show=True)
    FastMarkerCluster(coords).add_to(fg)
    fg.add_to(m)


def robust_vmin_vmax(values: List[Optional[float]]) -> tuple[float, float]:
    """Cálculo robusto de límites de color a partir de percentiles."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(arr, 2))
    vmax = float(np.nanpercentile(arr, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def annotate_values_in_geojson(geojson_obj: Dict, vals_map: Dict[str, float], key_field: str) -> Dict:
    """Inserta una propiedad 'value' por feature usando el diccionario código→valor (ligero, sin merges)."""
    from copy import deepcopy
    gj = deepcopy(geojson_obj)  # no mutar el cache
    for feat in gj.get("features", []):
        code = str(feat["properties"].get(key_field))
        feat["properties"]["value"] = vals_map.get(code, None)
    return gj


@st.cache_data(show_spinner=True)
def load_lines(path: str, simplify_tol_m: float | None = None) -> gpd.GeoDataFrame:
    """
    Carga líneas (GeoJSON/SHP). Devuelve GeoDataFrame en WGS84.
    Si simplify_tol_m se define, simplifica topológicamente con tolerancia en metros.
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El archivo {path} está vacío.")

    # Asegura 4326 (CRS:84 ~ EPSG:4326)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        try:
            epsg = gdf.crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            gdf = gdf.to_crs(4326)

    # Filtra geometrías válidas y sólo líneas
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if not any(t.startswith("Line") for t in gdf.geom_type.unique()):
        raise TypeError(f"Se esperaban líneas. Tipos: {gdf.geom_type.unique()}")

    # Arreglo de válidas sólo si hace falta
    if hasattr(gdf.geometry, "is_valid"):
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Simplificación en metros (proyecta a 3857 -> simplifica -> vuelve)
    if simplify_tol_m and simplify_tol_m > 0:
        gdf_m = gdf.to_crs(3857)
        gdf_m["geometry"] = gdf_m.geometry.simplify(simplify_tol_m, preserve_topology=True)
        gdf = gdf_m.to_crs(4326)

    return gdf


def add_road_layer(
    gdf_lines: gpd.GeoDataFrame,
    m: folium.Map,
    layer_name: str = "Carreteras",
    color_by: str = "highway",
    line_weight_by_cat: dict | None = None,
    color_map: dict | None = None,
    opacity: float = 0.9,
    tooltip_cols: list[str] | None = None,
):
    """
    Añade una capa de carreteras como un único GeoJson con estilo por categoría (p. ej. 'highway').
    """
    # Colores por tipo OSM por defecto
    color_map = color_map or {
        "motorway": "#e31a1c",
        "trunk": "#ff7f00",
        "primary": "#ffbf00",
        "secondary": "#1f78b4",
        "tertiary": "#33a02c",
        "residential": "#6a3d9a",
        "unclassified": "#b15928",
        "service": "#aaaaaa",
        "secondary_link": "#1f78b4",
        "tertiary_link": "#33a02c",
        "primary_link": "#ffbf00",
    }
    # Grosor por tipo
    line_weight_by_cat = line_weight_by_cat or {
        "motorway": 4.5,
        "trunk": 4.0,
        "primary": 3.5,
        "secondary": 3.0,
        "secondary_link": 2.6,
        "tertiary": 2.4,
        "tertiary_link": 2.2,
        "residential": 2.0,
        "unclassified": 2.0,
        "service": 1.5,
    }

    # Campos de tooltip
    tooltip_cols = tooltip_cols or [c for c in ["id", "provincia", "highway"] if c in gdf_lines.columns]
    tooltip = None
    if tooltip_cols:
        tooltip = GeoJsonTooltip(fields=tooltip_cols, aliases=tooltip_cols, sticky=False)

    # Para performance: sólo columnas necesarias
    keep = list(set(["geometry"] + tooltip_cols + [color_by]))
    gdf_small = gdf_lines[keep].copy()

    # GeoJSON único con estilo dinámico
    def _style_fn(feat):
        cat = str(feat["properties"].get(color_by, ""))
        color = color_map.get(cat, "#555555")
        weight = line_weight_by_cat.get(cat, 2.0)
        return {"color": color, "weight": weight, "opacity": opacity}

    fg = folium.FeatureGroup(name=layer_name, show=True)
    folium.GeoJson(
        data=gdf_small.__geo_interface__,
        style_function=_style_fn,
        tooltip=tooltip,
        name=layer_name,
    ).add_to(fg)
    fg.add_to(m)


@st.cache_data(show_spinner=True)
def load_as_points(path: str, polygon_method: str = "representative") -> gpd.GeoDataFrame:
    """
    Carga cualquier capa (Point/Polygon/LineString/Multiples) y devuelve SOLO puntos en WGS84.
    - Points -> tal cual
    - Polygons/MultiPolygons -> representative_point() (o centroid si polygon_method='centroid')
    - LineStrings/MultiLineStrings -> centroid
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El archivo {path} está vacío.")

    # CRS -> WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Limpieza básica
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    # Reparar inválidas si existieran
    if hasattr(gdf.geometry, "is_valid"):
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Convertir todo a puntos
    geom_type = gdf.geom_type.astype(str)
    is_point = geom_type.str.startswith("Point")
    is_poly = geom_type.str.startswith("Polygon")
    is_line = geom_type.str.startswith("Line")

    out = gdf.copy()
    # Polígonos -> punto representativo (dentro) o centroid
    if polygon_method == "centroid":
        out.loc[is_poly, "geometry"] = out.loc[is_poly, "geometry"].centroid
    else:
        out.loc[is_poly, "geometry"] = out.loc[is_poly, "geometry"].representative_point()

    # Líneas -> centroid (rápido y suficiente para “punto etiquetable”)
    out.loc[is_line, "geometry"] = out.loc[is_line, "geometry"].centroid

    # Filtrar solo puntos tras la conversión
    out = out[out.geometry.notnull() & ~out.geometry.is_empty]
    out = out[out.geom_type.astype(str).str.startswith("Point")]

    if out.empty:
        raise ValueError("No se pudieron derivar puntos de la capa proporcionada.")

    return out


# ================== Helpers para puntos categóricos (sin cluster) ==================
def _pick_name_col(gdf, candidates=("CENTRO_EDU", "Nombre", "NOMBRE", "name", "Tipo Institución", "Categoria")):
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def _build_palette(categories):
    # paleta de 20 colores (se reutiliza si hay más categorías)
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#5254a3", "#6b6ecf", "#9c9ede", "#8ca252", "#b5cf6b",
    ]
    cats = list(categories)
    return {cat: base[i % len(base)] for i, cat in enumerate(cats)}


def _add_categorical_legend(m, title, color_map):
    # Añade una leyenda HTML fija (esquina inferior derecha)
    items_html = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #333;"></span>'
        f'<span style="font-size:12px;color:#333;">{label}</span></div>'
        for label, color in color_map.items()
    )
    legend_html = f"""
    <div style="
        position: absolute; 
        bottom: 20px; right: 20px; 
        z-index: 9999; 
        background: rgba(255,255,255,0.92); 
        padding: 10px 12px; 
        border: 1px solid #ccc; 
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        ">
        <div style="font-weight:600;margin-bottom:6px;font-size:13px;color:#333;">{title}</div>
        {items_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def add_categorical_point_layer(
    gdf_points: gpd.GeoDataFrame,
    m: folium.Map,
    category_field: str,
    layer_name: str,
    radius: int = 4,
    fill_opacity: float = 0.9,
    stroke_color: str = "#222222",
    stroke_weight: float = 0.5,
):
    # categorías presentes (sin NaN)
    cats = sorted({str(v) for v in gdf_points[category_field].dropna().unique()})
    cmap = _build_palette(cats)
    name_col = _pick_name_col(gdf_points)

    fg = folium.FeatureGroup(name=layer_name, show=True)
    for _, row in gdf_points.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cat_val = str(row.get(category_field, "Sin categoría"))
        color = cmap.get(cat_val, "#7f7f7f")
        tip = None
        if name_col and pd.notnull(row.get(name_col)):
            tip = f"{name_col}: {row.get(name_col)}<br>{category_field}: {cat_val}"
        else:
            tip = f"{category_field}: {cat_val}"
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=radius,
            color=stroke_color,
            weight=stroke_weight,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            tooltip=tip
        ).add_to(fg)
    fg.add_to(m)
    # leyenda
    _add_categorical_legend(m, f"{layer_name} · {category_field}", cmap)


def merge_indices_with_extras(base_path: str,
                              base_cats_config: Dict[str, List[str]],
                              extra_sources: Dict[str, List[Dict]]):
    """
    Fusiona JSON base (por default COD_UGED) con fuentes extra que pueden:
      - usar otra clave (key_field, p.ej. NOMB_UGEC)
      - seleccionar/renombrar variables (select: {nuevo_nombre: nombre_original})
    Devuelve:
      indices_merged: dict con TODAS las variables (incluyendo claves extra)
      cats_map: categoría -> lista de índices existentes
      index_codekey_map: índice -> clave de unión que usa (COD_UGED/NOMB_UGEC/…)
    """
    # ---------- Base ----------
    base = load_indices_json(base_path)
    if "COD_UGED" not in base:
        raise ValueError("El JSON base no contiene COD_UGED")

    indices_merged: Dict[str, list] = {}
    index_codekey_map: Dict[str, str] = {}

    # Copiamos todo lo del base
    for k, v in base.items():
        indices_merged[k] = v

    # Categorías iniciales con lo que exista en base
    cats_map: Dict[str, List[str]] = build_category_index_options(indices_merged, base_cats_config)

    # ---------- Fuentes extra ----------
    for cat, sources in extra_sources.items():
        for src in sources:
            data = load_indices_json(src["path"])
            key_field = src.get("key_field") or "COD_UGED"  # default distrito si no especifican

            # Robustez: si pidieron mal NOMB_UGED pero existe NOMB_UGEC, corrige
            if key_field not in data:
                if key_field == "NOMB_UGED" and "NOMB_UGEC" in data:
                    key_field = "NOMB_UGEC"
                else:
                    raise ValueError(f"La fuente extra no contiene el campo clave '{key_field}'. "
                                     f"Claves disponibles (muestra): {list(data.keys())[:10]}")

            # 0) Asegura que el vector clave del extra queda en indices_merged
            #    (esto permite a series_from_arraystyle_index hacer el join)
            if key_field not in indices_merged:
                indices_merged[key_field] = data[key_field]

            # 1) ¿Qué variables incorporamos?
            if "select" in src and isinstance(src["select"], dict):
                # select: {nuevo_nombre: nombre_original}
                var_pairs = list(src["select"].items())
            else:
                # si no se especifica select, cogemos todas salvo el key_field
                var_pairs = [(k, k) for k in data.keys() if k != key_field]

            # 2) Copia/renombra variables y registra su clave
            for new_name, orig_name in var_pairs:
                if orig_name not in data:
                    # Silencioso: si no existe la variable origen, la saltamos
                    continue

                vals = data[orig_name]
                keys = indices_merged[key_field]

                # --- Saneado de longitudes: recorta al mínimo ---
                n = min(len(vals), len(keys))
                if n == 0:
                    # nada que fusionar
                    continue
                if len(vals) != n:
                    vals = vals[:n]
                if len(keys) != n:
                    indices_merged[key_field] = keys[:n]  # por si el vector clave era más largo

                # Guarda la serie y mapea su clave
                indices_merged[new_name] = vals
                index_codekey_map[new_name] = key_field

                # Mete en categoría evitando duplicados
                lst = cats_map.setdefault(cat, [])
                if new_name not in lst:
                    lst.append(new_name)

    return indices_merged, cats_map, index_codekey_map

def ensure_property_in_geojson(geojson_obj, shp_path, needed_field, join_on):
            if not geojson_obj.get("features"):
                return geojson_obj
            # Si ya está, no hacemos nada
            if needed_field in geojson_obj["features"][0]["properties"]:
                return geojson_obj
            try:
                gdf = gpd.read_file(shp_path)
                gdf = gdf[[join_on, needed_field]].drop_duplicates()
                lookup = dict(zip(gdf[join_on], gdf[needed_field]))
                for f in geojson_obj["features"]:
                    key = f["properties"].get(join_on)
                    if key in lookup:
                        f["properties"][needed_field] = lookup[key]
            except Exception:
                pass
            return geojson_obj

@st.cache_data(show_spinner=False)
def load_canton_boundaries_geojson(
    shp_path: str,
    canton_field: str = "NOMB_UGEC",
    simplify_tol: float = 0.0006,   # ~60 m en grados
    to_lines: bool = True,
    min_area_km2: float = 0.0,      # filtra islitas si quieres
) -> dict:
    """
    Carga límites cantonales 'ligeros':
      - Lee SHP distrital (metros), reproyecta a EPSG:4326
      - Dissolve por cantón
      - Simplifica geometría (tolerancia en grados)
      - (opcional) convierte a líneas (boundary) para dibujar sólo contornos
      - Devuelve GeoJSON dict
    Cacheado por Streamlit.
    """
    gdf = gpd.read_file(shp_path)

    # Reproyección a WGS84 (Folium)
    try:
        epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    except Exception:
        epsg = None
    if epsg != 4326:
        # si el SHP está en CRTM05 (EPSG:5367), esto lo llevará a 4326
        gdf = gdf.to_crs(4326)

    # Dissolve por cantón
    if canton_field not in gdf.columns:
        raise ValueError(f"No existe el campo '{canton_field}' en el SHP.")
    gdf = gdf.dissolve(by=canton_field, as_index=False)

    # (Opcional) filtrar por área si el SHP traía islas minúsculas
    if min_area_km2 and "AREA_KM2" in gdf.columns:
        gdf = gdf[gdf["AREA_KM2"] >= min_area_km2].copy()

    # Simplificación (preserva topología)
    gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)

    # Convertir a líneas para aligerar aún más
    if to_lines:
        gdf["geometry"] = gdf.boundary

    # Exportar a GeoJSON dict (evita volver a leer archivo)
    gj = json.loads(gdf.to_json())
    return gj