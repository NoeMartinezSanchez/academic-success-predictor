import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.compose import ColumnTransformer
import requests
from io import BytesIO
import time

# Configuración de la página
st.set_page_config(
    page_title="PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LINEA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (optimizado para fondo negro/oscuro)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #A5D6A7;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: bold;
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 15px;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #81C784;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        font-weight: 700;
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .metric-card {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #66BB6A;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        color: #E8F5E8 !important;
        font-weight: 600;
        backdrop-filter: blur(5px);
    }
    
    .recommendation-box {
        background: rgba(139, 195, 74, 0.2) !important;
        border: 2px solid #8BC34A !important;
        padding: 1.8rem !important;
        border-radius: 12px !important;
        border-left: 6px solid #AED581 !important;
        margin: 1rem 0 !important;
        box-shadow: 0 6px 12px rgba(139, 195, 74, 0.4) !important;
        color: #F1F8E9 !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .recommendation-box h1,
    .recommendation-box h2,
    .recommendation-box h3,
    .recommendation-box h4,
    .recommendation-box h5,
    .recommendation-box h6,
    .recommendation-box p,
    .recommendation-box li,
    .recommendation-box span,
    .recommendation-box div,
    .recommendation-box ul,
    .recommendation-box ol {
        color: #F1F8E9 !important;
        font-weight: 600 !important;
    }
    
    .recommendation-box strong,
    .recommendation-box b {
        color: #DCEDC8 !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .risk-high { 
        background: rgba(244, 67, 54, 0.2) !important;
        border: 2px solid #F44336 !important;
        border-left: 6px solid #EF5350 !important;
        color: #FFEBEE !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .risk-high h1,
    .risk-high h2,
    .risk-high h3,
    .risk-high h4,
    .risk-high h5,
    .risk-high h6,
    .risk-high p,
    .risk-high li,
    .risk-high span,
    .risk-high div,
    .risk-high ul,
    .risk-high ol {
        color: #FFCDD2 !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    
    .risk-medium { 
        background: rgba(255, 152, 0, 0.2) !important;
        border: 2px solid #FF9800 !important;
        border-left: 6px solid #FFB74D !important;
        color: #FFF8E1 !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .risk-medium h1,
    .risk-medium h2,
    .risk-medium h3,
    .risk-medium h4,
    .risk-medium h5,
    .risk-medium h6,
    .risk-medium p,
    .risk-medium li,
    .risk-medium span,
    .risk-medium div,
    .risk-medium ul,
    .risk-medium ol {
        color: #FFECB3 !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    
    .risk-low { 
        background: rgba(76, 175, 80, 0.2) !important;
        border: 2px solid #4CAF50 !important;
        border-left: 6px solid #66BB6A !important;
        color: #E8F5E8 !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .risk-low h1,
    .risk-low h2,
    .risk-low h3,
    .risk-low h4,
    .risk-low h5,
    .risk-low h6,
    .risk-low p,
    .risk-low li,
    .risk-low span,
    .risk-low div,
    .risk-low ul,
    .risk-low ol {
        color: #C8E6C9 !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .rf-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: #000000;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 800;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.4);
        border: 1px solid #81C784;
        text-shadow: none;
    }
    
    .feature-importance-box {
        background: rgba(139, 195, 74, 0.18) !important;
        border: 2px solid #8BC34A !important;
        padding: 1.8rem !important;
        border-radius: 12px !important;
        border-left: 6px solid #AED581 !important;
        margin: 1rem 0 !important;
        box-shadow: 0 5px 10px rgba(139, 195, 74, 0.35) !important;
        color: #F1F8E9 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .feature-importance-box h1,
    .feature-importance-box h2,
    .feature-importance-box h3,
    .feature-importance-box h4,
    .feature-importance-box h5,
    .feature-importance-box h6,
    .feature-importance-box p,
    .feature-importance-box li,
    .feature-importance-box span,
    .feature-importance-box div,
    .feature-importance-box ul,
    .feature-importance-box ol {
        color: #E8F5E8 !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Estilos generales para fondo oscuro */
    .stApp {
        background-color: #121212;
    }
    
    .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    /* Mejora para bullets y listas en fondo oscuro */
    ul, ol {
        color: #E8F5E8 !important;
        font-weight: 600 !important;
        line-height: 1.7;
    }
    
    li {
        margin-bottom: 0.6rem;
        color: #E8F5E8 !important;
        font-weight: 600 !important;
    }
    
    /* Mejora para textos en general en fondo oscuro */
    p {
        color: #E0E0E0 !important;
        font-weight: 500 !important;
        line-height: 1.7;
    }
    
    /* Asegurar que todos los elementos mantengan colores claros */
    .recommendation-box * {
        color: #F1F8E9 !important;
    }
    
    .feature-importance-box * {
        color: #E8F5E8 !important;
    }
    
    .metric-card * {
        color: #E8F5E8 !important;
    }
    
    /* Selectores adicionales para máxima compatibilidad */
    .recommendation-box > * {
        color: #F1F8E9 !important;
        font-weight: 600 !important;
    }
    
    .feature-importance-box > * {
        color: #E8F5E8 !important;
        font-weight: 600 !important;
    }
    
    /* Efectos de hover para mejor interactividad */
    .recommendation-box:hover {
        box-shadow: 0 8px 16px rgba(139, 195, 74, 0.5) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 6px 12px rgba(76, 175, 80, 0.4);
        transform: translateY(-1px);
        transition: all 0.3s ease;
    }
    
    .feature-importance-box:hover {
        box-shadow: 0 7px 14px rgba(139, 195, 74, 0.45) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con badge RF
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-header"> PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LINEA</h1>
    <span class="rf-badge">Random forest optimizado</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Sistema inteligente mejorado** que predice la probabilidad de éxito en educación en línea con **Random Forest optimizado**.
  Precisión del **89.8%** | ROC-AUC de **0.898** | Modelo más interpretable y balanceado
""")

# Mapeos para las variables
MAPEOS = {
    'si_no': {'Sí': 1, 'No': 0},
    'sexo': {'Hombre': 0, 'Mujer': 1, 'Otro': 2},
    'genero': {'Femenino': 0, 'Masculino': 1, 'Transgénero': 2, 'No binario': 3, 'Otro': 4},
    'situacion_conyugal': {'Soltero(a)': 0, 'Unión libre': 1, 'Casado(a)': 2, 'Divorciado(a)': 3, 'Separado(a)': 4, 'Viudo(a)': 5},
    'calificacion': {'Excelente': 0, 'Bueno': 1, 'Regular': 2, 'Malo': 3},
    'regimen_secundaria': {'Pública': 0, 'Privada': 1},
    'tipo_secundaria': {'General': 0, 'Técnica': 1, 'Telesecundaria': 2, 'Abierta': 3, 'Para adultos': 4},
    'edad_categoria': {'14-18': 0, '19-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
}




@st.cache_resource
def cargar_modelo_desde_url():
    """
    Cargar el modelo desde una URL pública (Dropbox/OneDrive) - CON DEBUG
    """
    try:
        # URL CORREGIDA
        dropbox_url = "https://dl.dropboxusercontent.com/scl/fi/myo7f1nfm001p8nfk35ps/modelo_exito_academico_RF_optimizado.pkl?rlkey=azkx43l6hmqzsz9f2zgi85bps&st=frca2m34&dl=1"
        
        st.sidebar.info("🌐 Descargando modelo desde la nube... (180 MB, puede tardar)")
        
        # Descargar el modelo con timeout aumentado
        response = requests.get(dropbox_url, timeout=180)
        response.raise_for_status()
        
        # ✅ DEBUG: Verificar qué se está descargando
        content = response.content
        st.sidebar.info(f"📦 Tamaño descargado: {len(content)} bytes")
        
        # Verificar si es un archivo válido (los primeros bytes de pickle)
        if len(content) > 10:
            st.sidebar.info(f"🔍 Primeros bytes: {content[:10]}")
        
        # Verificar si es HTML (podría ser página de error)
        if b'<html>' in content[:1000].lower() or b'<!doctype' in content[:1000].lower():
            st.sidebar.error("❌ Se descargó una página HTML, no el archivo")
            # Mostrar parte del contenido para debug
            st.sidebar.text(f"Contenido: {str(content[:500])}")
            return crear_modelo_demo(), {"modo_demo": True}
        
        st.sidebar.info("📦 Procesando modelo descargado...")
        
        # Cargar el modelo desde los bytes descargados
        modelo = pickle.load(BytesIO(content))
        
        # ✅ VERIFICACIÓN DESPUÉS de la carga completa
        if hasattr(modelo, 'predict_proba') and hasattr(modelo, 'predict'):
            st.sidebar.success("✅ Modelo verificado correctamente")
            
            metadata = {
                "roc_auc": 0.898,
                "accuracy": 82.5,
                "model_type": "Random Forest Optimizado"
            }
            
            return modelo, metadata
        else:
            st.sidebar.error("❌ El archivo descargado no es un modelo válido")
            return crear_modelo_demo(), {"modo_demo": True}
        
    except pickle.UnpicklingError as e:
        st.sidebar.error(f"❌ Error de formato pickle: {str(e)}")
        return crear_modelo_demo(), {"modo_demo": True}
        
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando modelo: {str(e)}")
        return crear_modelo_demo(), {"modo_demo": True}





        
def crear_formulario():
    """Crear formulario interactivo completo"""
    with st.sidebar:
        st.markdown("### FORMULARIO")
        st.markdown("Modelo **más balanceado** y **menos sesgado** por edad")
        
        with st.form("formulario_estudiante"):
            # ===== SECCIÓN 1: DATOS DEMOGRÁFICOS =====
            st.markdown('<div class="section-header">👤 Datos demográficos</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                edad = st.slider("Edad", 14, 70, 25, help="Edad actual del estudiante")
            with col2:
                sexo = st.selectbox("Sexo", options=list(MAPEOS['sexo'].keys()))
            
            genero = st.selectbox("Género", options=list(MAPEOS['genero'].keys()))
            situacion_conyugal = st.selectbox("Situación conyugal", options=list(MAPEOS['situacion_conyugal'].keys()))
            
            # ===== SECCIÓN 2: SALUD Y ORIGEN =====
            st.markdown('<div class="section-header"> Salud y origen</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            with col3:
                discapacidad = st.radio("¿Tiene discapacidad?", options=list(MAPEOS['si_no'].keys()))
            with col4:
                indigena = st.radio("¿Se considera indígena?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 3: SITUACIÓN ECONÓMICA =====
            st.markdown('<div class="section-header"> Situación económica</div>', unsafe_allow_html=True)
            st.info("💡 **RF Insight**: Los ingresos son 14x más importantes que en otros modelos")
            
            trabaja = st.radio("¿Trabaja actualmente?", options=list(MAPEOS['si_no'].keys()))
            
            if trabaja == 'Sí':
                horas_trabajo = st.slider("Horas de trabajo semanales", 0, 60, 40, 
                                        help=" Factor clave en Random Forest")
            else:
                horas_trabajo = 0
            
            ingresos_hogar = st.select_slider(
                " Ingresos mensuales del hogar (MXN)",
                options=[3000, 7500, 12500, 17500, 22500, 30000],
                value=12500,
                format_func=lambda x: f"${x:,.0f}",
                help=" Variable muy importante en RF"
            )
            
            beca = st.radio("¿Recibe alguna beca?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 4: TRAYECTORIA ACADÉMICA =====
            st.markdown('<div class="section-header"> Trayectoria académica</div>', unsafe_allow_html=True)
            
            col5, col6 = st.columns(2)
            with col5:
                regimen_secundaria = st.radio("Régimen de secundaria", options=list(MAPEOS['regimen_secundaria'].keys()))
            with col6:
                tipo_secundaria = st.selectbox("Tipo de secundaria", options=list(MAPEOS['tipo_secundaria'].keys()),
                                             help=" Más importante en RF que en GB")
            
            estudios_previos = st.radio("¿Tiene estudios previos de bachillerato?", options=list(MAPEOS['si_no'].keys()))
            cursos_linea = st.radio("¿Ha tomado cursos en línea antes?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 5: HABILIDADES Y RECURSOS =====
            st.markdown('<div class="section-header"> Habilidades y recursos</div>', unsafe_allow_html=True)
            st.success("✅ **RF Advantage**: Mejor balance entre todos los recursos")
            
            col7, col8 = st.columns(2)
            with col7:
                recursos_tec = st.slider(" Recursos tecnológicos", 1, 5, 3, 
                                       help=" 4x más importante en RF")
            with col8:
                responsabilidades = st.slider(" Responsabilidades", 1, 7, 3,
                                            help="Balance trabajo-estudio")
            
            comunicacion = st.select_slider(" Habilidad de comunicación", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            evaluacion = st.select_slider(" Habilidad evaluación información", 
                                        options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            organizacion = st.select_slider(" Habilidad de organización", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            
            # Calcular categoría de edad automáticamente
            if edad <= 18:
                edad_categoria = '14-18'
            elif edad <= 25:
                edad_categoria = '19-25'
            elif edad <= 35:
                edad_categoria = '26-35'
            elif edad <= 45:
                edad_categoria = '36-45'
            else:
                edad_categoria = '45+'
            
            # Botón de enviar con estilo RF - CORREGIDO use_container_width
            submitted = st.form_submit_button(" Predecir con Random Forest", use_container_width=True)
            
            datos = {
                'edad': edad, 'sexo': sexo, 'genero': genero, 'situacion_conyugal': situacion_conyugal,
                'discapacidad': discapacidad, 'indigena': indigena, 'trabaja': trabaja,
                'horas_trabajo_numeric': horas_trabajo, 'ingresos_hogar_numeric': ingresos_hogar,
                'beca': beca, 'regimen_secundaria': regimen_secundaria, 'tipo_secundaria': tipo_secundaria,
                'estudios_previos_bachillerato': estudios_previos, 'cursos_linea_3anos': cursos_linea,
                'score_recursos_tecnologicos': recursos_tec, 'score_responsabilidades': responsabilidades,
                'comunicacion_escrita': comunicacion, 'evaluacion_informacion': evaluacion,
                'organizacion_plataforma': organizacion, 'edad_categoria': edad_categoria
            }
            
            return submitted, datos

def preprocesar_datos(datos):
    """Preprocesar datos para el modelo"""
    datos_procesados = {}
    
    for key, value in datos.items():
        if key in ['edad', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 
                  'score_recursos_tecnologicos', 'score_responsabilidades']:
            datos_procesados[key] = float(value)
        else:
            # Buscar en los mapeos correspondientes
            for mapeo_key, mapeo in MAPEOS.items():
                if value in mapeo:
                    datos_procesados[key] = mapeo[value]
                    break
            else:
                datos_procesados[key] = value
    
    return datos_procesados

def crear_dataframe_modelo(datos_procesados):
    """Crear DataFrame con la estructura que el modelo espera"""
    columnas_esperadas = [
        'edad', 'sexo', 'genero', 'situacion_conyugal', 'discapacidad', 'indigena',
        'trabaja', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 'beca',
        'regimen_secundaria', 'tipo_secundaria', 'estudios_previos_bachillerato',
        'cursos_linea_3anos', 'score_recursos_tecnologicos', 'score_responsabilidades',
        'comunicacion_escrita', 'evaluacion_informacion', 'organizacion_plataforma',
        'edad_categoria'
    ]
    
    df = pd.DataFrame(columns=columnas_esperadas)
    
    for columna in columnas_esperadas:
        if columna in datos_procesados:
            df[columna] = [datos_procesados[columna]]
        else:
            df[columna] = [0]  # Valor por defecto
    
    # Convertir a numpy array sin nombres de características
    return df.values

def crear_gauge_chart(probabilidad):
    """Crear gráfico tipo gauge para la probabilidad"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probabilidad * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Éxito (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 40], 'color': "#FFCDD2"},
                {'range': [40, 60], 'color': "#FFECB3"},
                {'range': [60, 80], 'color': "#C8E6C9"},
                {'range': [80, 100], 'color': "#A5D6A7"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    
    fig.update_layout(height=300)
    return fig

def mostrar_feature_importance_personalizada(datos_usuario):
    """Mostrar feature importance personalizada"""
    # Feature importance del Random Forest (basada en tus resultados)
    features_rf = {
        'edad': 0.548,
        'edad_categoria': 0.102,
        'estudios_previos_bachillerato': 0.051,
        'horas_trabajo_numeric': 0.050,
        'ingresos_hogar_numeric': 0.034,
        'score_recursos_tecnologicos': 0.033,
        'cursos_linea_3anos': 0.032,
        'tipo_secundaria': 0.027,
        'score_responsabilidades': 0.023
    }
    
    st.markdown('<div class="section-header"> Importancia de factores para el caso particular </div>', unsafe_allow_html=True)
    
    # Crear gráfico de barras
    fig_importance = px.bar(
        x=list(features_rf.values())[:6],
        y=list(features_rf.keys())[:6],
        orientation='h',
        title="Top 6 factores más importantes (Random Forest)",
        color=list(features_rf.values())[:6],
        color_continuous_scale='Greens'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Análisis personalizado
    st.markdown("**🔍 Análisis de tu perfil:**")
    
    if datos_usuario['edad'] > 30:
        st.markdown(f'<div class="feature-importance-box">📊 **Edad ({datos_usuario["edad"]} años)**: Factor dominante pero menos que en otros modelos (55% vs 80% en GB)</div>', unsafe_allow_html=True)
    
    if datos_usuario['ingresos_hogar_numeric'] < 15000:
        st.markdown('<div class="feature-importance-box">💰 **Ingresos**: Nivel económico bajo puede ser un factor de riesgo importante</div>', unsafe_allow_html=True)
    
    if datos_usuario['horas_trabajo_numeric'] > 30:
        st.markdown('<div class="feature-importance-box">⏰ **Carga laboral**: Muchas horas de trabajo pueden afectar el rendimiento académico</div>', unsafe_allow_html=True)

def mostrar_resultados(probabilidad, prediccion, datos_originales, metadata):
    """Mostrar resultados de manera interactiva mejorada"""
    # Determinar nivel de riesgo con RF
    if probabilidad < 0.35:
        nivel_riesgo = "MUY ALTO"
        color_clase = "risk-high"
        emoji = "🔴"
        confianza = "Baja"
    elif probabilidad < 0.5:
        nivel_riesgo = "ALTO"  
        color_clase = "risk-high"
        emoji = "🟠"
        confianza = "Media"
    elif probabilidad < 0.7:
        nivel_riesgo = "MEDIO"
        color_clase = "risk-medium"
        emoji = "🟡"
        confianza = "Buena"
    elif probabilidad < 0.85:
        nivel_riesgo = "BAJO"
        color_clase = "risk-low"
        emoji = "🟢"
        confianza = "Alta"
    else:
        nivel_riesgo = "MUY BAJO"
        color_clase = "risk-low"
        emoji = "✅"
        confianza = "Muy alta"
    
    # Mostrar métricas principales con mejor diseño
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilidad de éxito", f"{probabilidad:.1%}")
    
    with col2:
        resultado = "✅ ÉXITO PROBABLE" if prediccion == 1 else "⚠️ RIESGO ALTO"
        st.metric("Predicción", resultado)
    
    with col3:
        st.metric("Nivel de riesgo", f"{emoji} {nivel_riesgo}")
    
    with col4:
        st.metric("Confianza", confianza)
    
    # Gráfico gauge interactivo
    st.plotly_chart(crear_gauge_chart(probabilidad), use_container_width=True)
    
    # Mostrar feature importance personalizada
    mostrar_feature_importance_personalizada(datos_originales)
    
    # Recomendaciones mejoradas para RF
    generar_recomendaciones_rf(probabilidad, datos_originales)
    
    return probabilidad  # Devolver para usar en el expander

def generar_recomendaciones_rf(probabilidad, datos):
    """Generar recomendaciones específicas para Random Forest"""
    st.markdown('<div class="section-header">🌲 Recomendaciones Random Forest</div>', unsafe_allow_html=True)
    
    recomendaciones = []
    
    # Recomendaciones basadas en el modelo RF balanceado
    if probabilidad < 0.4:
        recomendaciones.append("**Intervención integral inmediata**: El modelo RF indica múltiples factores de riesgo")
        recomendaciones.append("**Contacto urgente con asesor académico** para plan personalizado")
    elif probabilidad < 0.6:
        recomendaciones.append("**Plan de mejora multifactor**: RF identifica áreas específicas de mejora")
        recomendaciones.append("**Programa de acompañamiento** personalizado recomendado")
    else:
        recomendaciones.append("**Perfil favorable**: RF predice alta probabilidad de éxito")
        recomendaciones.append("**Mantener estrategia actual** con seguimiento regular")
    
    # Recomendaciones específicas basadas en features importantes en RF
    if datos['edad'] > 35:
        recomendaciones.append("**Programa para adultos mayores**: Estrategias específicas para estudiantes maduros")
    
    if datos['ingresos_hogar_numeric'] < 12000:  # RF da más peso a ingresos
        recomendaciones.append("**Apoyo económico crítico**: RF identifica ingresos como factor clave")
        recomendaciones.append("**Solicitar beca o apoyo financiero** urgentemente")
    
    if datos['horas_trabajo_numeric'] > 35:  # RF considera más el trabajo
        recomendaciones.append("**Gestión tiempo-trabajo crítica**: RF muestra que es factor importante")
        recomendaciones.append("**Negociar flexibilidad laboral** si es posible")
    
    if datos['score_recursos_tecnologicos'] < 3:  # RF valora más los recursos
        recomendaciones.append("**Apoyo tecnológico prioritario**: RF da más peso a recursos digitales")
        recomendaciones.append("**Programa de equipamiento tecnológico** recomendado")
    
    if datos['estudios_previos_bachillerato'] == 'No':
        recomendaciones.append("**Nivelación académica intensiva**: Compensar falta de base educativa")
    
    if datos['cursos_linea_3anos'] == 'No':
        recomendaciones.append("**Entrenamiento en educación digital**: RF considera experiencia online")
    
    # Fortalezas identificadas por RF
    if datos['score_recursos_tecnologicos'] >= 4:
        recomendaciones.append("**Aprovechar fortaleza tecnológica**: RF identifica como ventaja competitiva")
    
    if datos['ingresos_hogar_numeric'] >= 20000:
        recomendaciones.append("**Estabilidad económica favorable**: RF predice menor riesgo financiero")
    
    if datos['estudios_previos_bachillerato'] == 'Sí':
        recomendaciones.append("**Base académica sólida**: RF valora experiencia educativa previa")
    
    # Mostrar recomendaciones
    for i, rec in enumerate(recomendaciones, 1):
        st.markdown(f'<div class="recommendation-box">**{i}.** {rec}</div>', unsafe_allow_html=True)

def main():
    """Función principal"""
    pipeline, metadata = cargar_modelo_desde_url()


    if pipeline is None:
        st.error("No se pudo cargar ningún modelo")
        return
    
    # Crear formulario en sidebar
    submitted, datos_usuario = crear_formulario()
    
    # Inicializar variables
    probabilidad = None
    prediccion = None
    
    # Área principal para resultados
    if submitted:
        try:
            with st.spinner('🌲 Random Forest analizando... Modelo más balanceado procesando datos...'):
                # Preprocesar datos
                datos_procesados = preprocesar_datos(datos_usuario)
                
                # Crear array numpy (sin nombres de características)
                X_nuevo = crear_dataframe_modelo(datos_procesados)
                
                # PREDICCIÓN
                probabilidad = pipeline.predict_proba(X_nuevo)[0, 1]
                prediccion = pipeline.predict(X_nuevo)[0]
                
            # Mostrar resultados
            st.success("🌲 ¡Predicción Random Forest completada exitosamente!")
            probabilidad = mostrar_resultados(probabilidad, prediccion, datos_usuario, {})
            
            # Información técnica
            with st.expander("🔧 Detalles técnicos del modelo Random Forest"):
                st.write(f"**🎯 Probabilidad exacta:** {probabilidad:.6f}")
                st.write(f"**⚖️ Umbral de clasificación:** 0.5")
                st.write(f"**🌲 Modelo:** Random Forest Optimizado")
                
                if 'modo_demo' in metadata:
                    st.warning("⚠️ Modo demostración - usando modelo simplificado")
                    st.write("**📈 ROC-AUC:** 0.850 (demo)")
                    st.write("**🎯 Accuracy:** 80.0% (demo)")
                else:
                    st.write(f"**📈 ROC-AUC:** {metadata.get('roc_auc', 0.898):.4f}")
                    st.write(f"**🎯 Accuracy:** {metadata.get('accuracy', 82.5):.1f}%")
                
                # Ventajas del RF
                st.info("""
                **🌲 Ventajas del Random Forest sobre Gradient Boosting:**
                - ✅ Menos sesgado por edad (55% vs 80%)
                - ✅ Mejor balance entre factores
                - ✅ Más interpretable y explicable
                - ✅ Mayor importancia a factores socioeconómicos
                - ✅ Más robusto a outliers
                """)
                
        except Exception as e:
            st.error(f"❌ Error en la predicción Random Forest: {str(e)}")
            st.info("ℹ️ Verifica que todos los campos estén completos correctamente.")
    
    # Información cuando no hay predicción
    else:
        st.info("""
        👈 **Complete el formulario** para obtener una predicción con Random Forest optimizado.
        
        ** Ventajas del modelo Random Forest:**
        -  **Menos sesgado por edad** (55% vs 80% en otros modelos)
        -  **Mayor peso a factores económicos** (ingresos 14x más importantes)  
        -  **Considera mejor el balance trabajo-estudio**
        -  **Valora más los recursos tecnológicos**
        -  **Interpreta mejor la experiencia educativa**
        """)
        
        # Comparación con GB
        st.markdown('<div class="section-header"> Random Forest vs Gradient Boosting</div>', unsafe_allow_html=True)
        
        comparison_data = {
            'Métrica': ['ROC-AUC', 'Accuracy', 'Sesgo por Edad', 'Interpretabilidad', 'Balance de Factores'],
            'Random Forest': ['0.898', '82.5%', 'Bajo (55%)', 'Alta', 'Excelente'],
            'Gradient Boosting': ['0.886', '81.7%', 'Alto (80%)', 'Media', 'Pobre']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Factores más importantes en RF
        st.markdown('<div class="section-header"> Factores Más Importantes (Random Forest)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔺 Factores Principales:**")
            st.markdown("• **Edad del estudiante**: 54.8%")
            st.markdown("• **Categoría de edad**: 10.2%") 
            st.markdown("• **Estudios previos**: 5.1%")
            st.markdown("• **Horas de trabajo**: 5.0%")
            st.markdown("• **Ingresos del hogar**: 3.4%")
        
        with col2:
            st.markdown("**🔺 Factores Secundarios:**")
            st.markdown("• **Recursos tecnológicos**: 3.3%")
            st.markdown("• **Cursos en línea**: 3.2%")
            st.markdown("• **Tipo de secundaria**: 2.7%")
            st.markdown("• **Responsabilidades**: 2.3%")
            st.markdown("• **Otros factores**: 10.0%")

if __name__ == "__main__":
    main()
