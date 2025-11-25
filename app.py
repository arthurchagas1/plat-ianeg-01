# app.py
# Monitor de Consumo/Carregamento de Energia por Região (ONS) — Streamlit
# Fonte: ONS Dados Abertos — Carga de Energia Mensal (CSV público)
# Licença: CC-BY ONS (cite o ONS quando reutilizar). Veja o dataset no portal.  # noqa

import io
import sys
import time
import math
import json
import textwrap
import datetime as dt
from typing import Tuple, List, Dict

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ========= Config do App =========
st.set_page_config(
    page_title="Monitor de Energia por Subsistema (ONS)",
    page_icon="⚡",
    layout="wide"
)

ONS_CSV_URL = (
    # Recurso CSV do conjunto "Carga de Energia Mensal" (ONS Dados Abertos)
    # (S3 público; o portal mostra este CSV como um dos recursos do dataset)
    "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_me/CARGA_MENSAL.csv"
)

# ========= Utilidades =========
def _requests_get_robusto(url: str, max_tries: int = 3, timeout: int = 30) -> bytes:
    """Baixa bytes com algumas tentativas e backoff simples."""
    last_err = None
    for i in range(max_tries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Falha ao baixar dados de {url}: {last_err}")

def _hours_in_month(year: int, month: int) -> int:
    """Horas no mês (considera transição de mês de maneira simples)."""
    first = dt.datetime(year, month, 1)
    if month == 12:
        nxt = dt.datetime(year + 1, 1, 1)
    else:
        nxt = dt.datetime(year, month + 1, 1)
    return int((nxt - first).total_seconds() // 3600)

def _try_read_csv(content: bytes) -> pd.DataFrame:
    """Tenta ler CSV com separador autodetectado e fallback em ; e ,."""
    # 1) tentativa: sep=None (sniff), engine='python'
    bio = io.BytesIO(content)
    try:
        return pd.read_csv(bio, sep=None, engine="python", encoding="utf-8")
    except Exception:
        pass
    # 2) sep=';'
    bio.seek(0)
    try:
        return pd.read_csv(bio, sep=";", encoding="utf-8")
    except Exception:
        pass
    # 3) sep=','
    bio.seek(0)
    try:
        return pd.read_csv(bio, sep=",", encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Não foi possível ler o CSV: {e}")

def _normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes de colunas prováveis -> ano, mes, subsistema, carga_mwmed.

    Suporta tanto esquemas "antigos" (colunas já separadas de ano/mes/subsistema/carga)
    quanto o esquema oficial mais recente do ONS para Carga de Energia Mensal, com
    colunas como:
      - id_subsistema / nom_subsistema
      - din_instante (YYYY-MM-DD HH:MM:SS)
      - val_cargaenergiamwmed
    """
    # limpeza básica de nomes: tirar espaços e jogar tudo para minúsculas
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=lambda c: c.lower())

    cols = set(df.columns)

    # ----- Caso 1: novo layout oficial (din_instante + val_cargaenergiamwmed) -----
    if "din_instante" in cols and "val_cargaenergiamwmed" in cols:
        g = df.copy()

        # datetime de referência mensal
        g["din_instante"] = pd.to_datetime(g["din_instante"], errors="coerce")
        g["ano"] = g["din_instante"].dt.year.astype("Int64")
        g["mes"] = g["din_instante"].dt.month.astype("Int64")

        # Subsistema: preferir o nome legível; se não tiver, usar o id
        subs_col = None
        if "nom_subsistema" in cols:
            subs_col = "nom_subsistema"
        elif "id_subsistema" in cols:
            subs_col = "id_subsistema"

        if subs_col is not None:
            g["subsistema"] = g[subs_col].astype(str).str.strip().str.upper()
        else:
            g["subsistema"] = pd.NA

        # Valor numérico em MWmed
        g["carga_mwmed"] = pd.to_numeric(g["val_cargaenergiamwmed"], errors="coerce")

        # Filtrar linhas válidas
        g = g.dropna(subset=["ano", "mes", "subsistema", "carga_mwmed"])
        g = g[(g["mes"] >= 1) & (g["mes"] <= 12)]
        return g

    # ----- Caso 2: layouts antigos/alternativos (heurística genérica) -----
    def _contains(col: str, keys: List[str]) -> bool:
        c = col.lower()
        return any(k in c for k in keys)

    ano_col = next((c for c in df.columns if _contains(c, ["ano", "year"])), None)
    mes_col = next((c for c in df.columns if _contains(c, ["mes", "mês", "month"])), None)
    subs_col = next((c for c in df.columns if _contains(c, ["subsistema", "subsis", "subsystem"])), None)

    # carga (MWmed) pode vir como "carga", "valor", "mwmed" etc.
    carga_candidates = [
        c for c in df.columns if _contains(c, ["carga", "mwmed", "valor", "mw_medio", "mwmedio"])
    ]
    # priorizar colunas numéricas
    cand_num = [c for c in carga_candidates if pd.api.types.is_numeric_dtype(df[c])]
    carga_col = cand_num[0] if cand_num else (carga_candidates[0] if carga_candidates else None)

    rename_map: Dict[str, str] = {}
    if ano_col:
        rename_map[ano_col] = "ano"
    if mes_col:
        rename_map[mes_col] = "mes"
    if subs_col:
        rename_map[subs_col] = "subsistema"
    if carga_col:
        rename_map[carga_col] = "carga_mwmed"

    df = df.rename(columns=rename_map)

    missing = [c for c in ["ano", "mes", "subsistema", "carga_mwmed"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas essenciais ausentes após normalização: "
            f"{missing}. Colunas disponíveis no CSV: {sorted(df.columns.tolist())}"
        )

    # tipos
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df["carga_mwmed"] = pd.to_numeric(df["carga_mwmed"], errors="coerce")
    # limpar subsistema
    df["subsistema"] = df["subsistema"].astype(str).str.strip().str.upper()

    # remover linhas inválidas
    df = df.dropna(subset=["ano", "mes", "subsistema", "carga_mwmed"])
    df = df[(df["mes"] >= 1) & (df["mes"] <= 12)]
    return df

@st.cache_data(ttl=24 * 3600)
def carregar_ons_mensal() -> pd.DataFrame:
    """Baixa e prepara o dataset do ONS (Carga Mensal por subsistema)."""
    content = _requests_get_robusto(ONS_CSV_URL)
    raw = _try_read_csv(content)
    df = _normaliza_colunas(raw)

    # adicionar coluna data (primeiro dia do mês)
    df["data"] = pd.to_datetime(
        df["ano"].astype(int).astype(str) + "-" + df["mes"].astype(int).astype(str) + "-01",
        format="%Y-%m-%d",
        errors="coerce"
    )

    # energia (GWh/mês) ~ MWmed * horas_do_mês
    df["horas_mes"] = df.apply(lambda r: _hours_in_month(int(r["ano"]), int(r["mes"])), axis=1)
    df["energia_gwh"] = (df["carga_mwmed"] * df["horas_mes"]) / 1000.0

    # ordenar
    df = df.sort_values(["subsistema", "data"]).reset_index(drop=True)
    return df

def yoy_percent(group: pd.DataFrame, value_col: str = "energia_gwh") -> pd.DataFrame:
    g = group.sort_values("data").copy()
    g["energia_gwh_lag12"] = g[value_col].shift(12)
    g["yoy_%"] = 100.0 * (g[value_col] / g["energia_gwh_lag12"] - 1.0)
    return g

def rolling_ma(group: pd.DataFrame, value_col: str = "energia_gwh", window: int = 3) -> pd.DataFrame:
    g = group.sort_values("data").copy()
    g[f"ma{window}"] = g[value_col].rolling(window=window, min_periods=1).mean()
    return g

# ========= UI =========
st.title("⚡ Monitor de Energia por Subsistema — Brasil (ONS)")
st.markdown(
    """
    Este painel usa o conjunto **Carga de Energia Mensal** do ONS (média mensal de carga em MWmed por subsistema).
    Convertendo MWmed × horas do mês ⇒ GWh/mês, obtemos uma proxy comparável de **consumo/carga** por região do SIN.
    *Fonte: ONS Dados Abertos (CSV do dataset “Carga de Energia Mensal”).*
    """
)

with st.sidebar:
    st.header("⚙️ Controles")
    st.caption("Dica: selecione poucos subsistemas para comparar tendências com clareza.")

    df = carregar_ons_mensal()
    subsistemas = sorted(df["subsistema"].unique().tolist())

    sel_subs = st.multiselect(
        "Subsistemas",
        options=subsistemas,
        default=subsistemas  # todos por padrão
    )

    min_date = df["data"].min()
    max_date = df["data"].max()

    sel_range = st.date_input(
        "Período",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    show_ma = st.checkbox("Mostrar média móvel (3 meses)", value=True)
    show_yoy = st.checkbox("Mostrar variação anual (YoY)", value=True)

    st.divider()
    st.caption("Exportar os dados filtrados")
    btn_download = st.button("Gerar CSV filtrado para download")

# aplicar filtros
df_f = df[df["subsistema"].isin(sel_subs)].copy()
start, end = pd.to_datetime(sel_range[0]), pd.to_datetime(sel_range[1])
df_f = df_f[(df_f["data"] >= start) & (df_f["data"] <= end)]

if df_f.empty:
    st.warning("Não há dados para o filtro selecionado. Ajuste subsistemas ou período.")
    st.stop()

# KPIs do último mês disponível por subsistema
last_month = df_f["data"].max()
df_last = df_f[df_f["data"] == last_month].copy()
df_last = df_last.groupby("subsistema", as_index=False)["energia_gwh"].sum()

# calcular YoY por subsistema
if show_yoy:
    df_yoy = (
        df_f.groupby("subsistema", group_keys=False)
        .apply(yoy_percent)
        .drop(columns=["energia_gwh_lag12"])
    )
else:
    df_yoy = df_f.copy()
    df_yoy["yoy_%"] = np.nan

# média móvel
if show_ma:
    df_ma = df_f.groupby("subsistema", group_keys=False).apply(rolling_ma)
else:
    df_ma = df_f.copy()
    df_ma["ma3"] = np.nan

# ====== Layout principal ======
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.subheader("Série temporal (GWh/mês)")
    fig_ts = px.line(
        df_f,
        x="data", y="energia_gwh",
        color="subsistema",
        labels={"data": "Data", "energia_gwh": "GWh/mês", "subsistema": "Subsistema"},
        title=None
    )
    # sobrepor média móvel
    if show_ma:
        for subs in df_ma["subsistema"].unique():
            tmp = df_ma[df_ma["subsistema"] == subs]
            fig_ts.add_scatter(
                x=tmp["data"], y=tmp["ma3"],
                mode="lines",
                name=f"{subs} — MA3",
                line=dict(dash="dot")
            )
    st.plotly_chart(fig_ts, use_container_width=True, theme="streamlit")

with col2:
    st.subheader("Área empilhada — participação")
    # normalizar participação por mês
    share = (df_f
             .groupby(["data", "subsistema"], as_index=False)["energia_gwh"].sum())
    fig_area = px.area(
        share, x="data", y="energia_gwh", color="subsistema",
        groupnorm="fraction",
        labels={"data": "Data", "energia_gwh": "Participação", "subsistema": "Subsistema"},
        title=None
    )
    st.plotly_chart(fig_area, use_container_width=True, theme="streamlit")

with col3:
    st.subheader(f"Último mês\n{last_month:%b/%Y}")
    for _, r in df_last.sort_values("energia_gwh", ascending=False).iterrows():
        st.metric(label=r["subsistema"], value=f"{r['energia_gwh']:.0f} GWh")

st.divider()
colA, colB = st.columns(2)

with colA:
    st.subheader("Ranking (média dos últimos 12 meses)")
    # média 12m por subsistema (janela dentro do filtro)
    last12 = df_f[df_f["data"] >= (df_f["data"].max() - pd.DateOffset(months=12))]
    rank = (last12.groupby("subsistema", as_index=False)["energia_gwh"]
            .mean().rename(columns={"energia_gwh": "gwh_12m_media"}))
    fig_bar = px.bar(
        rank.sort_values("gwh_12m_media", ascending=False),
        x="subsistema", y="gwh_12m_media",
        labels={"subsistema": "Subsistema", "gwh_12m_media": "GWh/mês (média 12m)"},
        title=None
    )
    st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

with colB:
    st.subheader("Variação anual (YoY, %)")
    fig_yoy = px.line(
        df_yoy,
        x="data", y="yoy_%", color="subsistema",
        labels={"data": "Data", "yoy_%": "YoY (%)", "subsistema": "Subsistema"},
        title=None
    )
    st.plotly_chart(fig_yoy, use_container_width=True, theme="streamlit")

st.divider()
st.subheader("Heatmap — intensidade mensal (GWh)")
# pivot p/ heatmap (subsistemas nas colunas, meses no eixo x)
heat = (df_f.pivot_table(index="subsistema", columns="data", values="energia_gwh", aggfunc="sum")
        .sort_index())
# para evitar NaN no heatmap
heat = heat.fillna(0.0)
fig_heat = px.imshow(
    heat,
    aspect="auto",
    labels=dict(x="Data", y="Subsistema", color="GWh/mês"),
    color_continuous_scale="Viridis",
)
st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")

# ========= Novas análises =========

st.divider()
colC, colD = st.columns(2)

# 1) Perfil sazonal médio
with colC:
    st.subheader("Perfil sazonal médio por subsistema")
    season = df_f.copy()
    season["mes_num"] = season["data"].dt.month
    season_mean = (season
                   .groupby(["subsistema", "mes_num"], as_index=False)["energia_gwh"]
                   .mean())
    fig_season = px.line(
        season_mean,
        x="mes_num", y="energia_gwh", color="subsistema",
        markers=True,
        labels={"mes_num": "Mês", "energia_gwh": "GWh/mês (média histórica)", "subsistema": "Subsistema"},
        title=None
    )
    st.plotly_chart(fig_season, use_container_width=True, theme="streamlit")

# 2) Distribuição da variação YoY
with colD:
    st.subheader("Distribuição da variação anual (YoY)")
    df_yoy_valid = df_yoy.dropna(subset=["yoy_%"])
    if df_yoy_valid.empty:
        st.info("Não há janelas suficientes para calcular YoY com o período selecionado.")
    else:
        fig_yoy_box = px.box(
            df_yoy_valid,
            x="subsistema", y="yoy_%", points="outliers",
            labels={"subsistema": "Subsistema", "yoy_%": "YoY (%)"},
            title=None
        )
        st.plotly_chart(fig_yoy_box, use_container_width=True, theme="streamlit")

st.divider()

# 3) Correlação entre subsistemas
st.subheader("Correlação entre subsistemas (carga mensal)")
corr_pivot = (df_f
              .pivot_table(index="data", columns="subsistema", values="energia_gwh", aggfunc="sum"))
# Se só tiver um subsistema selecionado, não dá para calcular correlação
if corr_pivot.shape[1] < 2:
    st.info("Selecione pelo menos dois subsistemas para visualizar correlações.")
else:
    corr_mat = corr_pivot.corr()
    fig_corr = px.imshow(
        corr_mat,
        x=corr_mat.columns,
        y=corr_mat.index,
        labels=dict(x="Subsistema", y="Subsistema", color="Correlação"),
        zmin=-1, zmax=1,
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True, theme="streamlit")

st.divider()
colE, colF = st.columns(2)

# 4) Volatilidade x nível de carga
with colE:
    st.subheader("Volatilidade x nível de carga (últimos 24 meses)")
    window_24 = df_f[df_f["data"] >= (df_f["data"].max() - pd.DateOffset(months=24))]
    stats_vol = (window_24
                 .groupby("subsistema")["energia_gwh"]
                 .agg(["mean", "std"])
                 .reset_index()
                 .rename(columns={"mean": "gwh_medio", "std": "gwh_std"}))
    if stats_vol.empty:
        st.info("Período selecionado é muito curto para calcular volatilidade em 24 meses.")
    else:
        fig_vol = px.scatter(
            stats_vol,
            x="gwh_medio", y="gwh_std", text="subsistema",
            labels={
                "gwh_medio": "GWh/mês (média, 24m)",
                "gwh_std": "Desvio padrão (GWh/mês, 24m)"
            },
            title=None
        )
        fig_vol.update_traces(textposition="top center")
        st.plotly_chart(fig_vol, use_container_width=True, theme="streamlit")

# 5) Top 5 meses de maior carga por subsistema
with colF:
    st.subheader("Top 5 meses de maior carga por subsistema")
    top5 = (df_f
            .sort_values("energia_gwh", ascending=False)
            .groupby("subsistema")
            .head(5)
            .sort_values(["subsistema", "energia_gwh"], ascending=[True, False]))
    top5_view = top5[["subsistema", "data", "energia_gwh"]].copy()
    top5_view["data"] = top5_view["data"].dt.strftime("%Y-%m")
    top5_view["energia_gwh"] = top5_view["energia_gwh"].round(1)
    st.dataframe(
        top5_view.rename(columns={"subsistema": "Subsistema", "data": "Mês", "energia_gwh": "GWh/mês"}),
        use_container_width=True
    )

# Exportar CSV filtrado
if btn_download:
    csv_bytes = df_f.sort_values(["subsistema", "data"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar CSV (recorte filtrado)",
        data=csv_bytes,
        file_name=f"ons_carga_mensal_filtrado_{dt.datetime.now():%Y%m%d-%H%M%S}.csv",
        mime="text/csv"
    )

st.caption(
    "Notas: (i) ‘Carga (MWmed)’ do ONS foi convertida para ‘Energia (GWh/mês)’ multiplicando pelas horas do mês. "
    "(ii) ONS pode revisar dados; consulte o portal para metadados e definições. "
    "(iii) Este painel é apenas informativo."
)

# Rodapé com fonte
with st.expander("📎 Fonte de dados e licença"):
    st.markdown(textwrap.dedent(f"""
    - **ONS Dados Abertos — Carga de Energia Mensal** (CSV público): {ONS_CSV_URL}  
    - Licença: CC-BY ONS. Cite o ONS e informe alterações quando reutilizar.
    """))
