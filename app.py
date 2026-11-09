import numpy as np
import pandas as pd
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Análise Demográfica e Outliers", page_icon="📊", layout="wide"
)

st.title("📊 Análise Demográfica: Filtros, Estatísticas e Outliers")
st.markdown(
    "Este aplicativo realiza o recorte por **Gênero** e **Composição"
    " Demográfica**, calcula estatísticas descritivas e detecta *outliers*"
    " (valores discrepantes)."
)

# 1. Carregamento dos Dados
@st.cache_data
def carregar_dados_exemplo():
  np.random.seed(42)
  n = 500
  dados = {
      "ID": range(1, n + 1),
      "Genero": np.random.choice(
          ["Feminino", "Masculino", "Outro"], size=n, p=[0.48, 0.48, 0.04]
      ),
      "Composicao_Demografica": np.random.choice(
          ["Urbana Central", "Urbana Periférica", "Rural"],
          size=n,
          p=[0.5, 0.3, 0.2],
      ),
      "Renda": np.random.lognormal(mean=8, sigma=0.8, size=n) * 100,
      "Idade": np.random.randint(18, 75, size=n),
  }
  return pd.DataFrame(dados)


st.sidebar.header("📁 Fonte de Dados")
opcao_dados = st.sidebar.radio(
    "Escolha a fonte:",
    ("Usar dados de exemplo", "Fazer upload de arquivo CSV"),
)

if opcao_dados == "Fazer upload de arquivo CSV":
  arquivo_upload = st.sidebar.file_uploader("Envie seu arquivo CSV", type=["csv"])
  if arquivo_upload is not None:
    df = pd.read_csv(arquivo_upload)
  else:
    st.info("Por favor, envie um arquivo CSV ou mude para os dados de exemplo.")
    st.stop()
else:
  df = carregar_dados_exemplo()

# 2. Configuração dos Recortes (Filtros)
st.sidebar.header("🔍 Recorte e Filtros")

colunas_categoricas = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if not colunas_categoricas:
  st.error("O DataFrame não possui colunas categóricas para os recortes.")
  st.stop()

opcoes_genero_padrao = [
    c for c in colunas_categoricas if "genero" in c.lower() or "sexo" in c.lower()
]
col_genero = st.sidebar.selectbox(
    "Selecione a coluna de Gênero:",
    colunas_categoricas,
    index=(
        colunas_categoricas.index(opcoes_genero_padrao[0])
        if opcoes_genero_padrao
        else 0
    ),
)

generos_disponiveis = df[col_genero].dropna().unique().tolist()
generos_selecionados = st.sidebar.multiselect(
    "Filtrar por Gênero:",
    options=generos_disponiveis,
    default=generos_disponiveis,
)

opcoes_demog_padrao = [
    c
    for c in colunas_categoricas
    if "demograf" in c.lower() or "regiao" in c.lower() or "local" in c.lower()
]
col_demografia = st.sidebar.selectbox(
    "Selecione a coluna de Composição Demográfica:",
    colunas_categoricas,
    index=(
        colunas_categoricas.index(opcoes_demog_padrao[0])
        if opcoes_demog_padrao
        else (1 if len(colunas_categoricas) > 1 else 0)
    ),
)

demografias_disponiveis = df[col_demografia].dropna().unique().tolist()
demografias_selecionadas = st.sidebar.multiselect(
    "Filtrar por Composição Demográfica:",
    options=demografias_disponiveis,
    default=demografias_disponiveis,
)

df_filtrado = df[
    df[col_genero].isin(generos_selecionados)
    & df[col_demografia].isin(demografias_selecionadas)
]

st.subheader("📋 Dados Filtrados")
st.write(
    f"Mostrando **{len(df_filtrado)}** de **{len(df)}** registros totais após o"
    " recorte."
)
st.dataframe(df_filtrado.head(10), use_container_width=True)

# 3. Escolha da Variável Numérica para Análise
colunas_numericas = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

if not colunas_numericas:
  st.warning("Não há colunas numéricas disponíveis para calcular estatísticas.")
  st.stop()

st.sidebar.header("📈 Variável de Análise")
col_analise = st.sidebar.selectbox(
    "Selecione a métrica numérica:", colunas_numericas
)

# 4. Cálculo de Estatísticas Descritivas (Média, Mediana, Quartis, IQR)
if len(df_filtrado) > 0:
  serie_dados = df_filtrado[col_analise].dropna()

  media = serie_dados.mean()
  mediana = serie_dados.median()
  q1 = serie_dados.quantile(0.25)
  q3 = serie_dados.quantile(0.75)
  iqr = q3 - q1

  limite_inferior = q1 - 1.5 * iqr
  limite_superior = q3 + 1.5 * iqr

  outliers = serie_dados[
      (serie_dados < limite_inferior) | (serie_dados > limite_superior)
  ]

  st.markdown("---")
  st.subheader(f"📊 Estatísticas Descritivas para: `{col_analise}`")

  col1, col2, col3, col4, col5 = st.columns(5)
  col1.metric("Média", f"{media:,.2f}")
  col2.metric("Mediana", f"{mediana:,.2f}")
  col3.metric("1º Quartil (Q1)", f"{q1:,.2f}")
  col4.metric("3º Quartil (Q3)", f"{q3:,.2f}")
  col5.metric("IQR (Q3 - Q1)", f"{iqr:,.2f}")

  st.markdown("---")

  # 5. Detecção e Visualização de Outliers
  st.subheader("🚨 Detecção de Outliers (Método IQR)")

  st.write(f"- **Limite Inferior:** `{limite_inferior:,.2f}`")
  st.write(f"- **Limite Superior:** `{limite_superior:,.2f}`")
  st.write(
      f"- **Total de outliers encontrados:** `{len(outliers)}` (de um total de"
      f" {len(serie_dados)} registros válidos)"
  )

  if len(outliers) > 0:
    with st.expander("Visualizar registros identificados como Outliers"):
      st.dataframe(
          df_filtrado.loc[outliers.index], use_container_width=True
      )
  else:
    st.success("Nenhum outlier detectado para este recorte com os filtros atuais!")

  st.markdown("---")
  st.subheader("📈 Visualização Gráfica")

  aba1, aba2 = st.tabs(["Boxplot", "Distribuição"])

  with aba1:
    import altair as alt

    st.markdown(
        "*O Boxplot abaixo ajuda a visualizar visualmente os limites e os"
        " outliers.*"
    )
    chart = (
        alt.Chart(df_filtrado)
        .mark_boxplot(extent=1.5)
        .encode(
            x=col_genero,
            y=col_analise,
            color=col_demografia,
            tooltip=[col_genero, col_demografia, col_analise],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

  with aba2:
    st.bar_chart(serie_dados.value_counts(bins=10).sort_index())

else:
  st.warning(
      "A seleção atual de filtros não retornou nenhum dado. Tente selecionar"
      " mais opções na barra lateral."
  )