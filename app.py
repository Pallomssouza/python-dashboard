import streamlit as st
import pandas as pd
import numpy as np

# Configuração da Página
st.set_page_config(page_title="Projeto Logística", layout="wide")

# 1. DADOS REAIS (Tabela do Projeto)
df = pd.DataFrame({
    'Capital': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Vitória', 'Brasília', 'Goiânia', 'Cuiabá', 'Campo Grande'],
    'Lat': [-23.5337, -22.9083, -19.9167, -20.3155, -15.7939, -16.6869, -15.5989, -20.4428],
    'Lon': [-46.6253, -43.1964, -43.9345, -40.3128, -47.8828, -49.2648, -56.0949, -54.6464],
    'Populacao': [11904961, 6730729, 2415872, 343000, 2996899, 1503256, 691875, 962883]
})
df['Demanda'] = df['Populacao'] * 0.01

# 2. MENU LATERAL
st.sidebar.header("Painel de Controle")
capitais_sel = st.sidebar.multiselect("Cidades Atendidas:", df['Capital'].tolist(), default=df['Capital'].tolist())

st.sidebar.subheader("Centros de Distribuição (CDs)")
cd_sp_on = st.sidebar.checkbox("Ativar CD São Paulo", value=True)
cd_df_on = st.sidebar.checkbox("Ativar CD Brasília", value=False)

# Parâmetros fixos
C_f, C_t, h, K = 500000.0, 0.05, 2.0, 500.0
cds = []
if cd_sp_on: cds.append({'nome': 'CD São Paulo', 'lat': -23.5337, 'lon': -46.6252})
if cd_df_on: cds.append({'nome': 'CD Brasília', 'lat': -15.7938, 'lon': -47.8827})

st.title("🚚 Dashboard Logístico: Análise de Custos")

if not cds or not capitais_sel:
    st.error("Selecione pelo menos um CD e uma Cidade no menu lateral.")
else:
    # 3. CÁLCULOS
    df_f = df[df['Capital'].isin(capitais_sel)].copy()
    c_transp = 0.0
    demandas_agregadas = {c['nome']: 0.0 for c in cds}

    for _, row in df_f.iterrows():
        # Distância Euclidiana simplificada
        dists = [np.sqrt((row['Lat']-c['lat'])**2 + (row['Lon']-c['lon'])**2) * 111 for c in cds]
        idx = np.argmin(dists)
        c_transp += row['Demanda'] * dists[idx] * C_t
        demandas_agregadas[cds[idx]['nome']] += row['Demanda']

    # Cálculo de Estoque (Fórmula de Wilson simplificada)
    c_estq = sum([ (np.sqrt((2 * K * d) / h) / 2) * h for d in demandas_agregadas.values() if d > 0])
    c_fixo = float(len(cds) * C_f)

    # 4. EXIBIÇÃO EM COLUNAS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Resumo Financeiro")
        res_df = pd.DataFrame([
            {"Componente": "Transporte", "Valor": c_transp},
            {"Componente": "Estoque", "Valor": c_estq},
            {"Componente": "Custos Fixos", "Valor": c_fixo},
            {"Componente": "TOTAL", "Valor": c_transp + c_estq + c_fixo}
        ])

        def format_real(v): return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        res_df['Valor'] = res_df['Valor'].apply(format_real)
        st.table(res_df.set_index('Componente'))

    with col2:
        st.subheader("📊 Gráfico de Custos")
        st.bar_chart(pd.DataFrame({
            'Custo': ['Transporte', 'Estoque', 'Fixo'],
            'R$': [c_transp, c_estq, c_fixo]
        }).set_index('Custo'))

    st.subheader("📍 Localização Geográfica dos CDs")
    mapa_df = pd.DataFrame([{'lat': c['lat'], 'lon': c['lon']} for c in cds])
    st.map(mapa_df)