import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
st.set_page_config(page_title="Logística Pro | Suporte à Decisão", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DE DADOS
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Capital': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Vitória', 'Brasília', 'Goiânia', 'Cuiabá', 'Campo Grande'],
        'lat': [-23.53, -22.90, -19.91, -20.31, -15.79, -16.68, -15.59, -20.44],
        'lon': [-46.62, -43.19, -43.93, -40.31, -47.88, -49.26, -56.09, -54.64],
        'Demanda_Media': [119049, 67307, 24158, 3430, 29968, 15032, 6918, 9628]
    })

df_base = load_data()
C_f, C_t, h, K, LT = 500000.0, 0.05, 2.0, 500.0, 2

# 3. BARRA LATERAL (SIDEBAR) - INTERFACE DE ENTRADA
with st.sidebar:
    st.title("⚙️ Parâmetros")
    st.markdown("### 1. Escopo de Atendimento")
    cidades_sel = st.multiselect("Selecione as cidades atendidas:", 
                                 df_base['Capital'].tolist(), 
                                 default=df_base['Capital'].tolist())
    
    st.markdown("---")
    st.markdown("### 2. Gestão de Risco")
    st.write("Selecione o nível de serviço desejado:")
    nivel_servico_pct = st.slider("", 80.0, 99.9, 95.0, help="Probabilidade de atendimento sem falta de estoque.")
    z_score = norm.ppf(nivel_servico_pct / 100)
    
    st.info(f"**Nível de Serviço: {nivel_servico_pct}%**\nIsso define o rigor do seu estoque de segurança para cobrir incertezas.")

# FILTRAGEM
df = df_base[df_base['Capital'].isin(cidades_sel)].copy()
df['Sigma'] = df['Demanda_Media'] * 0.20

# 4. ÁREA PRINCIPAL
if df.empty:
    st.warning("⚠️ Por favor, selecione ao menos uma cidade na barra lateral para iniciar a análise.")
else:
    st.title("🚚 Dashboard de Otimização Logística")
    st.caption("Ferramenta de suporte à decisão baseada em Simulação de Monte Carlo e Trade-offs de Redes.")

    # SIMULAÇÃO DOS 3 CENÁRIOS (SP, BSB e HÍBRIDO)
    cenarios = {
        "Centralização (CD São Paulo)": [{'nome': 'CD São Paulo', 'lat': -23.53, 'lon': -46.62}],
        "Centralização (CD Brasília)": [{'nome': 'CD Brasília', 'lat': -15.79, 'lon': -47.88}],
        "Descentralização (SP + BSB)": [
            {'nome': 'CD São Paulo', 'lat': -23.53, 'lon': -46.62},
            {'nome': 'CD Brasília', 'lat': -15.79, 'lon': -47.88}
        ]
    }

    def executar_simulacao(lista_cds, z):
        custos = []
        for _ in range(200):
            c_transp = 0
            agreg_d = {c['nome']: 0.0 for c in lista_cds}
            agreg_v = {c['nome']: 0.0 for c in lista_cds}
            for _, row in df.iterrows():
                dem = max(0, np.random.normal(row['Demanda_Media'], row['Sigma']))
                dists = [np.sqrt((row['lat']-c['lat'])**2 + (row['lon']-c['lon'])**2) * 111 for c in lista_cds]
                idx = np.argmin(dists)
                c_transp += dem * dists[idx] * C_t
                agreg_d[lista_cds[idx]['nome']] += dem
                agreg_v[lista_cds[idx]['nome']] += row['Sigma']**2
            c_estq = sum([(np.sqrt((2*K*d)/h)/2)*h + (z * np.sqrt(v) * np.sqrt(LT))*h 
                          for d,v in zip(agreg_d.values(), agreg_v.values()) if d > 0])
            custos.append(c_transp + c_estq + (len(lista_cds) * C_f))
        return custos

    with st.spinner('Analisando cenários e calculando trade-offs...'):
        stats = {nome: {'raw': executar_simulacao(lista, z_score)} for nome, lista in cenarios.items()}
        for nome in stats:
            stats[nome]['media'] = np.mean(stats[nome]['raw'])
            stats[nome]['desvio'] = np.std(stats[nome]['raw'])

    melhor_cenario = min(stats, key=lambda k: stats[k]['media'])

    # --- SEÇÃO DE RECOMENDAÇÃO ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([1.2, 2])
    
    with res_col1:
        st.success(f"### 🎯 Decisão Sugerida:\n**{melhor_cenario}**")
        st.metric("Custo Médio Estimado", f"R$ {stats[melhor_cenario]['media']:,.2f}")
        st.metric("Risco Operacional (Desvio)", f"R$ {stats[melhor_cenario]['desvio']:,.2f}")
        st.write("A recomendação prioriza o menor custo total integrado (Transporte + Estoque + Instalação).")

    with res_col2:
        st.subheader("📍 Visualização da Malha Sugerida")
        cd_mapa = pd.DataFrame([{'lat': c['lat'], 'lon': c['lon']} for c in cenarios[melhor_cenario]])
        st.map(pd.concat([df[['lat', 'lon']], cd_mapa]), color="#ff4b4b", zoom=3)

    # --- SEÇÃO DE ANÁLISE TÉCNICA ---
    st.markdown("---")
    st.subheader("📊 Análise Comparativa e Sensibilidade")
    
    tab1, tab2 = st.tabs(["💰 Comparativo de Custos", "📈 Sensibilidade e Risco"])
    
    with tab1:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            df_plot = pd.DataFrame({"Custo Total (R$)": [v['media'] for v in stats.values()]}, index=stats.keys())
            st.bar_chart(df_plot)
        with col_c2:
            st.write("**Resumo por Estratégia**")
            st.dataframe(df_plot.style.format("R$ {:,.2f}"))

    with tab2:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Custo vs Nível de Serviço**")
            # Gráfico de Sensibilidade
            z_range = [1.28, 1.64, 2.33] # 90%, 95%, 99%
            c_sens = [np.mean(executar_simulacao(cenarios[melhor_cenario], z)) for z in z_range]
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(["90%", "95%", "99%"], c_sens, marker='o', color='#2ecc71', linewidth=2)
            ax1.set_ylabel("Custo Total (R$)")
            st.pyplot(fig1)
        
        with g2:
            st.info("""
            **Por que a sensibilidade importa?**
            Este gráfico mostra que aumentar o Nível de Serviço gera um aumento exponencial no custo de estoque. 
            Uma malha **robusta** é aquela onde a curva de custo não sofre saltos bruscos diante da incerteza.
            """)
            st.markdown(fr"**Impacto do Risco:** {(stats[melhor_cenario]['desvio']/stats[melhor_cenario]['media'])*100:.2f}% de volatilidade.")

    # --- RODAPÉ E EXPORTAÇÃO ---
    st.markdown("---")
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        with st.expander("📚 Fundamentação Teórica (Risk Pooling e Robustez)"):
            st.write(fr"""
            A simulação utiliza o conceito de **Risk Pooling**, onde a centralização de estoques reduz a variabilidade total ($\sigma_{{total}} = \sqrt{{\sum \sigma_i^2}}$). 
            O modelo compara se a redução no custo fixo e no estoque de segurança da centralização supera a eficiência de frete da descentralização.
            """)
    with f_col2:
        st.download_button("📥 Exportar Simulação (CSV)", 
                           pd.DataFrame(stats[melhor_cenario]['raw']).to_csv(index=False).encode('utf-8'), 
                           "resultado_logistica.csv")