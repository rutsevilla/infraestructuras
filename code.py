# app.py
import json
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st
from functions import *

import folium
from folium import GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium
import branca.colormap as cm

# ================== RUTAS (ajústalas) ==================
SHP_PATH = "./data/UGED_MGN_2022_simplified.shp"  # distritos (CRS en metros)
CARRETERAS_GEOJSON_PATH = "./data/geojson-proces/carreteras-principales-cr.geojson"
CENTROS_GEOJSON_PATH = "./data/geojson-proces/centros-educativos-cr.geojson"
HOSPITALES_GEOJSON_PATH = "./data/geojson-proces/centros-salud-cr.geojson"
POLICIA_GEOJSON_PATH   = "./data/geojson-proces/cuerpos-seguridad-cr.geojson"
INDICES_JSON_PATH = "./data/dades-proces/indices-desarrollo-cr.json"
DELITOS_JSON_PATH   = "./data/dades-proces/delitos-cr.json"
EDU_EXTRA_JSON_PATH = "./data/dades-proces/educacion-cr.json"
BUS_GEOJSON_PATH       = "./data/geojson-proces/bus-paradas-cr.geojson"
EMPRESAS_GEOJSON_PATH = "./data/geojson-proces/empresas-oficios-cr.geojson"
TRANSITO_JSON_PATH   = "./data/dades-proces/accidentes-trafico-cr.json"
logo_path = "./static/logos/TDP_Logo_White.svg"

# Clave de unión en el shapefile (código de distrito)
DISTRICT_KEY_FIELD = "COD_UGED"

# Mapa de categorías → lista de índices (puedes añadir más por categoría)
CATS_CONFIG = {
    "Salud": ["ids_salud"],
    "Participación Electoral": ["ids_participacion"],
    "Seguridad": ["ids_seguridad"],
    "Educación": ["ids_educacion"],
    "Economía": ["ids_economia"],
}
# Cada entrada añade TODAS las variables del JSON a la categoría indicada.
EXTRA_SOURCES = {
    "Seguridad": [
        {"path": DELITOS_JSON_PATH,
        "key_field": "COD_UGED"}    # <- importante
    ],
    "Educación": [
        {"path": EDU_EXTRA_JSON_PATH,
        "key_field": "COD_UGED"}    # <- importante
    ],
    # NUEVO: categoría Transito con selección/renombrado y clave NOMB_UGEC
    "Transito": [
        {
            "path": TRANSITO_JSON_PATH,
            "key_field": "NOMB_UGEC"               # <- importante
            # # selecciona y renombra columnas del JSON a nombres “bonitos”
            # "select": {
            #     "accidentes_trafico" : "%_accidentes_con_victimas_2024",
            #     "accidentes_materiales": "%_accidentes_con_materiales_2024"
            # }
        }
    ],
}

# ================== CONFIG STREAMLIT ==================
st.set_page_config(
    page_title="Infraestructuras Costa Rica",
    page_icon="./static/logos/TDP-circle-white.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  ================== ESTILOS ===================
st.markdown(f"""
<style>
/* ===== Tipografía: local (./static) con fallback a Google ===== */
@font-face {{
  font-family: 'PoppinsLocal';
  src: url('./static/Poppins-Regular.woff2') format('woff2'),
       url('./static/Poppins-Regular.ttf') format('truetype');
  font-weight: 300;
  font-style: normal;
  font-display: swap;
}}

/* ===== Fondo y contenedor principal ===== */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(90deg, #175CA1, #07A9E0 140%);
  background-attachment: fixed;
}}

/* ===== Cabecera (logo + título) ===== */
.header-row {{ display:flex; align-items:center; gap:12px; }}
.header-row h1 {{ margin:0; font-size:4vh; font-weight:500; color:#fff; }}
.header-row img {{ height:5vh; width:auto; }}


/* ===== Ajustes generales ===== */
.block-container label:empty {{ margin:0; padding:0; }}
footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ display:none !important; }}
MainMenu {{ visibility: hidden; }}
main blockquote, .block-container {{ padding-top: 0.6rem; padding-bottom: 0.6rem; }}

/* ======================================================================================= */
/* ===== A PARTIR D'AQUÍ AFEGIU ELS ESTILS DEL GRAFICS Y ELEMENS QUE ANEU CONSTRUINT ===== */
/* ======================================================================================= */

</style>
""", unsafe_allow_html=True)

# ======================================= CUERPO APP =======================================

logo_data_uri = img_to_data_uri(logo_path)   #logo

# ================== CABECERA ==================
st.markdown(
    f"""
    <div class="header-box">
      <div class="header-row">
        <img src="{logo_data_uri}" alt="TDP Logo" />
        <h1>Infraestructuras Costa Rica</h1>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
# ================== CONTENIDO PRINCIPAL ==================
left, right = st.columns([1, 3])

with left:
    with st.container(border=True):
        st.markdown("<p style='font-size: 2.5vh; font-weight: 300; color: #fff;'>Capas</p>", unsafe_allow_html=True)
        chk_centros    = st.checkbox("Centros educativos", value=False)
        chk_hospital   = st.checkbox("Centros de Salud", value=False)
        chk_policia    = st.checkbox("Estaciones de policía", value=False)
        chk_carreteras = st.checkbox("Carreteras", value=False)
        chk_bus        = st.checkbox("Paradas de bus", value=False)
        chk_empresas   = st.checkbox("Empresas y oficios", value=False)

    with st.container(border=True):
        st.markdown("<p style='font-size: 2.5vh; font-weight: 300; color: #fff;'>Selección de variable</p>", unsafe_allow_html=True)

        # Fusiona JSON base + extras (delitos, educación extra, etc.)
        indices_data, cats_map, index_codekey_map = merge_indices_with_extras(
            INDICES_JSON_PATH, CATS_CONFIG, EXTRA_SOURCES
        )


        # Añadimos opción "Ninguna" en categoría e índice
        cat_options = ["—"] + list(cats_map.keys())
        cat = st.selectbox("Categoría", cat_options, index=0)

        if cat == "— ":
            index_options = ["—"]
        else:
            index_options = ["—"] + cats_map.get(cat, [])

        idx_label = st.selectbox("Variable", index_options, index=0, disabled=(cat == "—"))
        idx_name = None if idx_label.startswith("—") else idx_label  # None si no se selecciona índice real


with right:
    with st.container(border=True):
        # ================== Distritos (GeoJSON simplificado + cacheado) ==================
        simplify_tol = 0.0008  # fijo (~80 m de tolerancia)
        geojson_dist_raw, bounds = load_districts_geojson_simplified(SHP_PATH, DISTRICT_KEY_FIELD, simplify_tol)
        # Fijamos bounds y centro (Costa Rica)
        minx, miny, maxx, maxy = [-86, 8, -82, 11.5]
        center_lat, center_lon = 9.7, -84.1

        # ================== Índices ==================
        use_index = idx_label is not None and idx_label != "—"
        if use_index:
            idx_name = idx_label
            # clave de unión para este índice: "COD_UGED" (distrito) o "NOMB_UGEC" (cantón), etc.
            code_key_for_idx = index_codekey_map.get(idx_name, DISTRICT_KEY_FIELD)
            s_vals = series_from_arraystyle_index(indices_data, idx_name, code_key=code_key_for_idx)
            vals_map = s_vals.astype(float).to_dict()
            vmin, vmax = robust_vmin_vmax(list(vals_map.values()))
        else:
            idx_name = None
            vals_map = {}
            vmin, vmax = 0.0, 1.0
            code_key_for_idx = DISTRICT_KEY_FIELD  # por defecto

        # ================== Mapa Folium ==================
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles="cartodbpositron",
            width="100%",
            control_scale=True
        )

        # Oculta atribuciones/enlaces
        m.get_root().html.add_child(folium.Element("""
        <style>
        .leaflet-control-attribution {display:none !important;}
        .leaflet-container a {display:none !important;}
        </style>
        """))

        # Colormap solo si hay índice
        colormap = None
        if use_index:
            colormap = cm.LinearColormap(
                colors=["#F5694C", "#F2EC79", "#4ECC54"],  # rojo -> amarillo -> verde
                vmin=vmin,
                vmax=vmax
            )
            colormap.caption = idx_name    

        # Si la clave de unión del índice es distinta del campo distrito por defecto,
        # rellenamos esa propiedad en el GeoJSON si faltara.
        if code_key_for_idx != DISTRICT_KEY_FIELD:
            geojson_dist_raw = ensure_property_in_geojson(
                geojson_dist_raw, SHP_PATH, needed_field=code_key_for_idx, join_on=DISTRICT_KEY_FIELD
            )

        # GeoJSON anotado con la clave correcta (distrito o cantón)
        if use_index:
            geojson_dist = annotate_values_in_geojson(geojson_dist_raw, vals_map, key_field=code_key_for_idx)
        else:
            geojson_dist = geojson_dist_raw

        # Estilo: neutro si no hay índice
        def style_fn(feature):
            if not use_index:
                return {"fillColor": "#ACE6FA", "color": "#777777", "weight": 0.7, "fillOpacity": 0.5}
            val = feature["properties"].get("value", None)
            fill = "#ACE6FA" if (val is None or (isinstance(val, float) and np.isnan(val))) else colormap(val)
            return {"fillColor": fill, "color": "#777777", "weight": 0.7, "fillOpacity": 0.7}

        # Tooltips (evita desajustes fields/aliases)
        tip_fields = []
        aliases = []

        # Siempre muestra código/ nombre de distrito si existen
        if DISTRICT_KEY_FIELD in geojson_dist["features"][0]["properties"]:
            tip_fields.append(DISTRICT_KEY_FIELD); aliases.append("Cod. distrito")
        if "NOMB_UGED" in geojson_dist["features"][0]["properties"]:
            tip_fields.append("NOMB_UGED"); aliases.append("Distrito")

        # Si existe info de cantón en el GeoJSON, añádela
        for cand_key, alias in [("COD_UGEC", "Cod. cantón"), ("NOMB_UGEC", "Cantón")]:
            if cand_key in geojson_dist["features"][0]["properties"]:
                tip_fields.append(cand_key); aliases.append(alias)

        # Finalmente, el valor del índice si aplica
        if use_index:
            tip_fields.append("value"); aliases.append(idx_name)

        GeoJson(
            data=geojson_dist,
            name="Distritos",
            style_function=style_fn,
            tooltip=GeoJsonTooltip(
                fields=tip_fields,
                aliases=aliases,
                sticky=False,
                smooth_factor=0.2
            )
        ).add_to(m)
    
        # ================== Añadir bordes cantonales (líneas gruesas) ==================
        try:
            # usa el helper cacheado
            gj_cantones = load_canton_boundaries_geojson(
                SHP_PATH,
                canton_field="NOMB_UGEC",
                simplify_tol=0.0006,  # ajusta (0.0004–0.001) según fluidez/precisión
                to_lines=True,        # dibujar sólo contorno (más ligero)
                min_area_km2=0.0
            )

            def canton_style_fn(_):
                return {
                    "color": "#4F4F4F",  # o "#FFFFFF" si quieres contraste sobre polígonos oscuros
                    "weight": 1,
                    "fill": False,
                    "fillOpacity": 0.0,
                    "opacity": 0.9,
                }

            folium.GeoJson(
                data=gj_cantones,
                name="Límites cantonales",
                style_function=canton_style_fn,
                tooltip=folium.GeoJsonTooltip(fields=["NOMB_UGEC"], aliases=["Cantón:"], sticky=False),
                overlay=True,
                control=True,
                show=True,
                smooth_factor=0.2
            ).add_to(m)

        except Exception as e:
            st.warning(f"No se pudieron añadir los límites cantonales: {e}")


        # ================== Capas de puntos (clusters) ==================
        if chk_centros:
            try:
                gdf_centros = load_points(CENTROS_GEOJSON_PATH)
                # Colorea por "Tipo Institución" (Privado/Público)
                field = "Tipo Institucion" if "Tipo Institucion" in gdf_centros.columns else None
                if field is None:
                    # fallback si el campo viniera con otra variante
                    for cand in ["Tipo_Institucion", "tipo_institucion", "tipo", "Tipo"]:
                        if cand in gdf_centros.columns:
                            field = cand
                            break
                if field is None:
                    st.warning("No se encontró el campo 'Tipo Institucion' en Centros educativos; se usarán marcadores grises.")
                    field = "__dummy__"
                    gdf_centros[field] = "Centro educativo"

                add_categorical_point_layer(
                    gdf_centros,
                    m,
                    category_field=field,
                    layer_name="Centros educativos",
                    radius=4
                )
            except Exception as e:
                st.warning(f"No se pudo cargar 'Centros educativos': {e}")

        if chk_hospital:
            try:
                gdf_hosp = load_points(HOSPITALES_GEOJSON_PATH)
                field = "Categoria" if "Categoria" in gdf_hosp.columns else None
                if field is None:
                    for cand in ["Categoría", "categoria", "CAT"]:
                        if cand in gdf_hosp.columns:
                            field = cand
                            break
                if field is None:
                    st.warning("No se encontró el campo 'Categoria' en Centros de salud; se usarán marcadores grises.")
                    field = "__dummy__"
                    gdf_hosp[field] = "Centro de salud"

                add_categorical_point_layer(
                    gdf_hosp,
                    m,
                    category_field=field,
                    layer_name="Centros de Salud",
                    radius=4
                )
            except Exception as e:
                st.warning(f"No se pudo cargar 'Centros de Salud': {e}")

        if chk_policia:
            try:
                # Convierte cualquier geometría a puntos (representative point para polígonos)
                gdf_pol_pts = load_as_points(POLICIA_GEOJSON_PATH, polygon_method="representative")
                field = "Categoria" if "Categoria" in gdf_pol_pts.columns else None
                if field is None:
                    for cand in ["Categoría", "categoria", "Tipo"]:
                        if cand in gdf_pol_pts.columns:
                            field = cand
                            break
                if field is None:
                    st.warning("No se encontró el campo 'Categoria' en Cuerpos de seguridad; se usarán marcadores grises.")
                    field = "__dummy__"
                    gdf_pol_pts[field] = "Cuerpo de seguridad"

                add_categorical_point_layer(
                    gdf_pol_pts,
                    m,
                    category_field=field,
                    layer_name="Cuerpos de seguridad",
                    radius=4
                )
            except Exception as e:
                st.warning(f"No se pudo cargar 'Policía': {e}")

        # ================== Carreteras (líneas) ==================
        if chk_carreteras:
            try:
                # Asegúrate de tener definida esta ruta antes (p.ej. en la sección de rutas)
                # CARRETERAS_GEOJSON_PATH = "/share/home/ruts/notebooks/costa-rica/infraestructuras/geojson-proces/carreteras-principales.geojson"
                gdf_roads = load_lines(CARRETERAS_GEOJSON_PATH, simplify_tol_m=8)  # simplificación opcional (metros)
                add_road_layer(
                    gdf_roads,
                    m,
                    layer_name="Carreteras",
                    color_by="highway"
                )
            except Exception as e:
                st.warning(f"No se pudo cargar 'Carreteras': {e}")
        
        # ================== Paradas de bus ==================
        # ================== Paradas de bus (puntos sin cluster) ==================
        if chk_bus:
            try:
                gdf_bus = load_points(BUS_GEOJSON_PATH)

                # Intenta usar un campo categórico si existe (ajusta la lista si tu GeoJSON usa otro nombre)
                field = None
                for cand in ["Tipo", "TIPO", "tipo", "Categoria", "Categoría", "categoria", "Clase", "CLASE"]:
                    if cand in gdf_bus.columns:
                        field = cand
                        break

                if field is not None:
                    # Colorear por categoría + leyenda
                    add_categorical_point_layer(
                        gdf_bus,
                        m,
                        category_field=field,
                        layer_name="Paradas de bus",
                        radius=4
                    )
                else:
                    # Sin campo categórico: marcadores simples (mismo color)
                    add_point_layer(
                        gdf_bus,
                        m,
                        layer_name="Paradas de bus",
                        radius=4,
                        color="#1D3557",
                        fill_color="#457B9D",
                        fill_opacity=0.9,
                        tooltip_col=next((c for c in ["Nombre", "NOMBRE", "name"] if c in gdf_bus.columns), None)
                    )

            except Exception as e:
                st.warning(f"No se pudo cargar 'Paradas de bus': {e}")


        # ================== Empresas y oficios (puntos sin cluster, color por 'grupo_es') ==================
        if chk_empresas:
            try:
                gdf_empresas = load_points(EMPRESAS_GEOJSON_PATH)

                field = "grupo_es" if "grupo_es" in gdf_empresas.columns else None
                if field is not None:
                    # Colorear por categoría + leyenda
                    add_categorical_point_layer(
                        gdf_empresas,
                        m,
                        category_field=field,
                        layer_name="Empresas y oficios",
                        radius=4
                    )
                else:
                    # Sin campo categórico: marcadores simples (mismo color)
                    add_point_layer(
                        gdf_empresas,
                        m,
                        layer_name="Empresas y oficios",
                        radius=4,
                        color="#8B0000",
                        fill_color="#E63946",
                        fill_opacity=0.9,
                        tooltip_col=next((c for c in ["Nombre", "NOMBRE", "name"] if c in gdf_empresas.columns), None)
                    )

            except Exception as e:
                st.warning(f"No se pudo cargar 'Empresas y oficios': {e}")

        # Leyenda solo si hay índice
        if colormap is not None:
            colormap.add_to(m)

        st_folium(m, height=600, use_container_width=True, returned_objects=[])
