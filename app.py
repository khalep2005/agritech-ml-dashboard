import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN INICIAL Y CARGA DEL MODELO
# ============================================================================

st.set_page_config(
    page_title="AgriTech IoT Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el modelo pre-entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_cultivos.pkl')

modelo = cargar_modelo()

# ============================================================================
# ESTILOS PERSONALIZADOS (DARK MODE FRIENDLY)
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c853 0%, #64dd17 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_gauge_chart(valor, titulo, rango_min, rango_max, unidad, color):
    """Crea un gráfico tipo velocímetro (gauge) para sensores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=valor,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{titulo}</b>", 'font': {'size': 20}},
        number={'suffix': f" {unidad}", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [rango_min, rango_max], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [rango_min, (rango_max-rango_min)*0.33 + rango_min], 'color': 'rgba(100, 221, 23, 0.2)'},
                {'range': [(rango_max-rango_min)*0.33 + rango_min, (rango_max-rango_min)*0.66 + rango_min], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [(rango_max-rango_min)*0.66 + rango_min, rango_max], 'color': 'rgba(244, 67, 54, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': valor
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def crear_radar_chart(N, P, K):
    """Crea un gráfico de radar comparando NPK actual vs ideal"""
    # Valores ideales promedio (estos son aproximados, ajusta según tu dominio)
    valores_ideales = {
        'Nitrógeno': 70,
        'Fósforo': 75,
        'Potasio': 100
    }
    
    categorias = ['Nitrógeno (N)', 'Fósforo (P)', 'Potasio (K)']
    valores_actuales = [N, P, K]
    valores_ideal = [valores_ideales['Nitrógeno'], valores_ideales['Fósforo'], valores_ideales['Potasio']]
    
    fig = go.Figure()
    
    # Valores actuales
    fig.add_trace(go.Scatterpolar(
        r=valores_actuales,
        theta=categorias,
        fill='toself',
        name='Valores Actuales',
        line=dict(color='#00c853', width=3),
        fillcolor='rgba(0, 200, 83, 0.3)'
    ))
    
    # Valores ideales
    fig.add_trace(go.Scatterpolar(
        r=valores_ideal,
        theta=categorias,
        fill='toself',
        name='Valores Ideales',
        line=dict(color='#ff6b6b', width=3, dash='dash'),
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 150],
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=80, r=80, t=40, b=80)
    )
    
    return fig

def crear_barras_probabilidades(clases, probabilidades, top_n=5):
    """Crea un gráfico de barras horizontal con las probabilidades"""
    top_indices = np.argsort(probabilidades)[::-1][:top_n]
    top_clases = [clases[i].capitalize() for i in top_indices]
    top_probs = [probabilidades[i] * 100 for i in top_indices]
    
    # Colores degradados
    colors = ['#00c853', '#64dd17', '#aeea00', '#ffd600', '#ffab00']
    
    fig = go.Figure(go.Bar(
        x=top_probs,
        y=top_clases,
        orientation='h',
        marker=dict(
            color=colors[:len(top_probs)],
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        text=[f'{p:.1f}%' for p in top_probs],
        textposition='outside',
        textfont=dict(size=14, color='white')
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Top 5 Cultivos Recomendados</b>',
            font=dict(size=18, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Probabilidad (%)',
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            title='',
            color='white'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    return fig

def generar_valores_aleatorios():
    """Genera valores aleatorios realistas para simular sensores IoT"""
    return {
        'N': np.random.randint(20, 120),
        'P': np.random.randint(10, 130),
        'K': np.random.randint(10, 190),
        'temp': round(np.random.uniform(12.0, 38.0), 1),
        'hum': round(np.random.uniform(30.0, 95.0), 1),
        'ph': round(np.random.uniform(4.5, 8.5), 1),
        'rain': round(np.random.uniform(30.0, 250.0), 1)
    }

# ============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

if 'valores_sensores' not in st.session_state:
    st.session_state.valores_sensores = {
        'N': 50, 'P': 50, 'K': 50, 'temp': 25.0,
        'hum': 70.0, 'ph': 6.5, 'rain': 100.0
    }

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================

st.markdown('<h1 class="main-header">🌾 AgriTech IoT Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Empresarial de Monitoreo en Tiempo Real y Recomendación Inteligente de Cultivos</p>', unsafe_allow_html=True)

# Timestamp y estado del sistema
col_time1, col_time2, col_time3 = st.columns([2, 1, 1])
with col_time1:
    st.caption(f"🕐 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_time2:
    st.caption("🟢 Sistema Operativo")
with col_time3:
    st.caption(f"📡 Sensores: 7/7")

st.divider()

# ============================================================================
# BOTÓN DE SIMULACIÓN IoT (PROMINENTE)
# ============================================================================

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🔄 SIMULAR LECTURA DE SENSORES EN CAMPO", use_container_width=True):
        st.session_state.valores_sensores = generar_valores_aleatorios()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SECCIÓN 1: GAUGES DE SENSORES PRINCIPALES (TEMPERATURA Y HUMEDAD)
# ============================================================================

st.markdown("### � Monitoreo de Sensores en Tiempo Real")

gauge_col1, gauge_col2 = st.columns(2)

with gauge_col1:
    fig_temp = crear_gauge_chart(
        st.session_state.valores_sensores['temp'],
        "Temperatura Ambiente",
        8, 44, "°C", "#ff6b6b"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with gauge_col2:
    fig_hum = crear_gauge_chart(
        st.session_state.valores_sensores['hum'],
        "Humedad Relativa",
        14, 100, "%", "#4fc3f7"
    )
    st.plotly_chart(fig_hum, use_container_width=True)

st.divider()

# ============================================================================
# SECCIÓN 2: PANEL DE CONTROL DE SENSORES Y ANALÍTICA
# ============================================================================

col_left, col_right = st.columns([1.2, 1.5])

# --- PANEL IZQUIERDO: CONTROL DE SENSORES ---
with col_left:
    st.markdown("### ⚙️ Panel de Control de Sensores")
    
    with st.container():
        st.markdown("#### 🌱 Nutrientes del Suelo")
        N = st.slider(
            "Nitrógeno (N)",
            0, 140,
            st.session_state.valores_sensores['N'],
            key='slider_N',
            help="Nivel de nitrógeno en el suelo (mg/kg)"
        )
        P = st.slider(
            "Fósforo (P)",
            5, 145,
            st.session_state.valores_sensores['P'],
            key='slider_P',
            help="Nivel de fósforo en el suelo (mg/kg)"
        )
        K = st.slider(
            "Potasio (K)",
            5, 205,
            st.session_state.valores_sensores['K'],
            key='slider_K',
            help="Nivel de potasio en el suelo (mg/kg)"
        )
        ph = st.slider(
            "pH del Suelo",
            3.5, 10.0,
            st.session_state.valores_sensores['ph'],
            key='slider_ph',
            help="Nivel de acidez/alcalinidad del suelo"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 🌤️ Condiciones Climáticas")
        temp = st.slider(
            "Temperatura (°C)",
            8.0, 44.0,
            st.session_state.valores_sensores['temp'],
            key='slider_temp',
            help="Temperatura ambiente en grados Celsius"
        )
        hum = st.slider(
            "Humedad Relativa (%)",
            14.0, 100.0,
            st.session_state.valores_sensores['hum'],
            key='slider_hum',
            help="Porcentaje de humedad en el aire"
        )
        rain = st.slider(
            "Precipitación (mm)",
            20.0, 300.0,
            st.session_state.valores_sensores['rain'],
            key='slider_rain',
            help="Precipitación acumulada en milímetros"
        )
    
    # Actualizar session state con valores de sliders
    st.session_state.valores_sensores.update({
        'N': N, 'P': P, 'K': K, 'temp': temp,
        'hum': hum, 'ph': ph, 'rain': rain
    })

# --- PANEL DERECHO: ANALÍTICA Y PREDICCIÓN ---
with col_right:
    st.markdown("### 🧠 Motor de Inteligencia Artificial")
    
    # Crear dataframe con los datos
    datos_entrada = pd.DataFrame(
        [[N, P, K, temp, hum, ph, rain]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )
    
    # Obtener predicción y probabilidades
    prediccion = modelo.predict(datos_entrada)[0]
    probabilidades = modelo.predict_proba(datos_entrada)[0]
    confianza_maxima = np.max(probabilidades) * 100
    
    # KPIs principales
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.metric(
            label="🎯 Cultivo Óptimo",
            value=prediccion.upper(),
            delta="Recomendado",
            delta_color="normal"
        )
    
    with kpi_col2:
        st.metric(
            label="📈 Confianza del Modelo",
            value=f"{confianza_maxima:.1f}%",
            delta=f"+{confianza_maxima-50:.1f}%" if confianza_maxima > 50 else f"{confianza_maxima-50:.1f}%",
            delta_color="normal" if confianza_maxima > 50 else "inverse"
        )
    
    with kpi_col3:
        # Calcular un "score de salud del suelo" basado en NPK
        health_score = min(100, int((N + P + K) / 3.5))
        st.metric(
            label="🌿 Salud del Suelo",
            value=f"{health_score}/100",
            delta="Óptimo" if health_score > 70 else "Revisar",
            delta_color="normal" if health_score > 70 else "inverse"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico de barras con probabilidades
    clases = modelo.classes_
    fig_barras = crear_barras_probabilidades(clases, probabilidades, top_n=5)
    st.plotly_chart(fig_barras, use_container_width=True)
    
    # Insights de negocio
    with st.expander("💡 Insights y Recomendaciones", expanded=False):
        st.markdown(f"""
        **Análisis del Modelo Random Forest:**
        - El algoritmo ha procesado **7 variables ambientales** simultáneamente
        - Nivel de confianza: **{confianza_maxima:.1f}%**
        - Cultivo principal: **{prediccion.capitalize()}**
        
        **Factores Clave Detectados:**
        - Temperatura: {'✅ Óptima' if 15 <= temp <= 35 else '⚠️ Fuera de rango ideal'}
        - Humedad: {'✅ Adecuada' if 40 <= hum <= 90 else '⚠️ Revisar niveles'}
        - pH: {'✅ Balanceado' if 5.5 <= ph <= 7.5 else '⚠️ Ajustar acidez'}
        
        **Recomendación:** {prediccion.capitalize()} es el cultivo más adecuado para las condiciones actuales del campo.
        """)

st.divider()

# ============================================================================
# SECCIÓN 3: GRÁFICO RADAR NPK
# ============================================================================

st.markdown("### 🎯 Análisis Comparativo de Nutrientes (NPK)")

col_radar1, col_radar2 = st.columns([2, 1])

with col_radar1:
    fig_radar = crear_radar_chart(N, P, K)
    st.plotly_chart(fig_radar, use_container_width=True)

with col_radar2:
    st.markdown("#### 📋 Resumen de Nutrientes")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(0,200,83,0.1) 0%, rgba(100,221,23,0.1) 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00c853;'>
        <p style='margin: 0; font-size: 0.9rem; color: #aaa;'>Nitrógeno (N)</p>
        <p style='margin: 0; font-size: 1.8rem; font-weight: bold; color: #00c853;'>{N} mg/kg</p>
        <hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0;'>
        <p style='margin: 0; font-size: 0.9rem; color: #aaa;'>Fósforo (P)</p>
        <p style='margin: 0; font-size: 1.8rem; font-weight: bold; color: #64dd17;'>{P} mg/kg</p>
        <hr style='border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0;'>
        <p style='margin: 0; font-size: 0.9rem; color: #aaa;'>Potasio (K)</p>
        <p style='margin: 0; font-size: 1.8rem; font-weight: bold; color: #aeea00;'>{K} mg/kg</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Indicador de balance NPK
    balance_npk = abs(N - 70) + abs(P - 75) + abs(K - 100)
    if balance_npk < 50:
        st.success("✅ Balance NPK excelente")
    elif balance_npk < 100:
        st.warning("⚠️ Balance NPK aceptable")
    else:
        st.error("❌ Balance NPK requiere ajustes")

st.divider()

# ============================================================================
# FOOTER
# ============================================================================

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🤖 Modelo:** Random Forest Classifier")
with footer_col2:
    st.markdown(f"**📊 Precisión:** ~{confianza_maxima:.0f}%")
with footer_col3:
    st.markdown("**🔬 Variables:** 7 sensores IoT")

st.caption("© 2026 AgriTech Analytics | Dashboard Empresarial de Agricultura Inteligente")