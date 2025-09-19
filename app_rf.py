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

# Configuración de la página
st.set_page_config(
    page_title="PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LINEA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS universales compatibles con todos los navegadores
st.markdown("""
<style>
    /* ===== VARIABLES DE COLOR (FUNCIONAN EN CLARO/OSCURO) ===== */
    :root {
        --primary-color: #2563eb;       /* Azul profesional */
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --success-color: #059669;       /* Verde éxito */
        --warning-color: #d97706;       /* Amarillo advertencia */
        --danger-color: #dc2626;        /* Rojo peligro */
        --text-primary: #1f2937;        /* Texto oscuro */
        --text-secondary: #4b5563;      /* Texto secundario */
        --bg-light: #f8fafc;            /* Fondo claro */
        --bg-card: #ffffff;             /* Fondo tarjetas */
        --border-color: #e5e7eb;        /* Bordes suaves */
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* ===== ESTILOS BASE UNIVERSALES ===== */
    .stApp {
        background-color: var(--bg-light) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stMarkdown, .stText, .stWrite {
        color: var(--text-primary) !important;
        line-height: 1.6;
    }

    /* ===== HEADERS Y TÍTULOS ===== */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color) !important;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem;
    }

    .section-header {
        font-size: 1.4rem;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }

    /* ===== COMPONENTES DE LA APLICACIÓN ===== */
    .metric-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    .recommendation-box {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }

    .risk-high { 
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        border-left: 4px solid var(--danger-color) !important;
    }

    .risk-medium { 
        background: #fffbeb !important;
        border: 1px solid #fed7aa !important;
        border-left: 4px solid var(--warning-color) !important;
    }

    .risk-low { 
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-left: 4px solid var(--success-color) !important;
    }

    /* ===== BADGES Y ELEMENTOS ESPECIALES ===== */
    .rf-badge {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white !important;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: none;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .feature-importance-box {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }

    /* ===== SIDEBAR (COMPATIBLE CON MODO CLARO/OSCURO) ===== */
    .css-1d391kg, .css-1y4p8pa {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    .css-1d391kg p, .css-1y4p8pa p, 
    .css-1d391kg label, .css-1y4p8pa label,
    .css-1d391kg .stMarkdown, .css-1y4p8pa .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* ===== FORMULARIOS Y CONTROLES ===== */
    .stSelectbox, .stSlider, .stRadio, .stNumberInput {
        background-color: var(--bg-card) !important;
    }

    .stSelectbox div, .stSlider div, .stRadio div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    /* ===== TEXTO Y TIPOGRAFÍA ===== */
    .recommendation-box h1,
    .recommendation-box h2,
    .recommendation-box h3,
    .recommendation-box h4,
    .recommendation-box h5,
    .recommendation-box h6,
    .recommendation-box p,
    .recommendation-box li,
    .recommendation-box span,
    .recommendation-box div {
        color: var(--text-primary) !important;
    }

    .recommendation-box strong,
    .recommendation-box b {
        color: var(--primary-color) !important;
    }

    /* ===== EFECTOS HOVER ===== */
    .metric-card:hover {
        transform: translateY(-2px);
        transition: all 0.2s ease;
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .recommendation-box:hover {
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }

    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .section-header {
            font-size: 1.2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }

    /* ===== ESTILOS ESPECÍFICOS PARA STREAMLIT ===== */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* ===== SCROLLBAR PERSONALIZADO ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }

    /* ===== ESTILOS PARA TABLAS ===== */
    .stDataFrame, .stTable {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* ===== ESTILOS PARA EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
    }

    /* ===== CORRECCIÓN PARA TEXTO EN SIDEBAR ===== */
    .css-1y4p8pa .stMarkdown {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    .css-1y4p8pa .stRadio label {
        color: var(--text-primary) !important;
    }

    /* ===== CONTRASTE MEJORADO PARA ACCESIBILIDAD ===== */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }

    .stSuccess {
        background-color: #f0fdf4 !important;
        border-left: 4px solid var(--success-color) !important;
    }

    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid var(--warning-color) !important;
    }

    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid var(--danger-color) !important;
    }

    /* ===== ICONOS Y ELEMENTOS GRÁFICOS ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
    }

</style>
""", unsafe_allow_html=True)

# Título principal con badge RF
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-header">PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LÍNEA</h1>
    <span class="rf-badge">Random Forest Optimizado</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Sistema inteligente mejorado** que predice la probabilidad de éxito en educación en línea con **Random Forest optimizado**.
Precisión del **89.8%** | ROC-AUC de **0.898** | Modelo más interpretable y balanceado
""")




@st.cache_resource
def cargar_modelo():
    """
    Función de carga robusta con workaround de compatibilidad
    """
    try:
        # WORKAROUND PARA COMPATIBILIDAD - Esto resuelve el error
        try:
            # Intentar importar la clase faltante
            from sklearn.compose._column_transformer import _RemainderColsList
        except ImportError:
            # Si no existe, crearla dinámicamente
            class _RemainderColsList(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            
            # Inyectar la clase en el módulo
            import sklearn.compose._column_transformer
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
            
            # También en ColumnTransformer por si acaso
            from sklearn.compose import ColumnTransformer
            ColumnTransformer._RemainderColsList = _RemainderColsList
        
        # AHORA intentar cargar el modelo
        if os.path.exists('modelo_rf_streamlit_compatible.joblib'):
            pipeline = joblib.load('modelo_rf_streamlit_compatible.joblib')
            metadata = joblib.load('metadatos_compatible.joblib')
            
            if hasattr(pipeline, 'predict'):
                st.sidebar.success("✅ Modelo cargado con workaround de compatibilidad")
                return pipeline, metadata
        
        # Si joblib falla, intentar con pickle
        if os.path.exists('modelo_rf_streamlit_compatible.pkl'):
            with open('modelo_rf_streamlit_compatible.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            with open('metadatos_compatible.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            if hasattr(pipeline, 'predict'):
                st.sidebar.success("✅ Modelo cargado con pickle")
                return pipeline, metadata
                
    except Exception as e:
        st.error(f"❌ Error en carga: {str(e)}")
    
    return None, None




# Mapeos para las variables (iguales que antes)
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
            
            # Botón de enviar con estilo RF
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
    
    return df

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
    pipeline, metadata = cargar_modelo()
    if pipeline is None:
        return
    
    # Mostrar información del modelo cargado
    if metadata:
        st.success(f"✅ Modelo Random Forest cargado - Entrenado el {metadata['fecha_entrenamiento'].strftime('%d/%m/%Y')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 ROC-AUC", f"{metadata['roc_auc']:.3f}")
        with col2:
            st.metric("📊 Accuracy", f"{metadata['accuracy']:.3f}")
        with col3:
            st.metric("🌲 Estimadores", metadata['parametros'].get('n_estimators', 'N/A'))
    
    # Crear formulario en sidebar
    submitted, datos_usuario = crear_formulario()
    
    # Área principal para resultados
    if submitted:
        try:
            with st.spinner('🌲 Random Forest analizando... Modelo más balanceado procesando datos...'):
                # Preprocesar datos
                datos_procesados = preprocesar_datos(datos_usuario)
                
                # Crear DataFrame
                X_nuevo = crear_dataframe_modelo(datos_procesados)
                
                # Hacer predicción
                probabilidad = pipeline.predict_proba(X_nuevo)[0, 1]
                prediccion = pipeline.predict(X_nuevo)[0]
                
            # Mostrar resultados
            st.success("🌲 ¡Predicción Random Forest completada exitosamente!")
            mostrar_resultados(probabilidad, prediccion, datos_usuario, metadata)
            
            # Información técnica
            with st.expander("🔧 Detalles técnicos del modelo Random Forest"):
                st.write(f"**🎯 Probabilidad exacta:** {probabilidad:.6f}")
                st.write(f"**⚖️ Umbral de clasificación:** 0.5")
                st.write(f"**🌲 Modelo:** Random Forest Optimizado")
                st.write(f"**📈 ROC-AUC:** {metadata['roc_auc']:.4f}")
                st.write(f"**🎯 Accuracy:** {metadata['accuracy']:.4f}")
                
                # Parámetros del modelo
                st.write("**⚙️ Parámetros:**")
                params_importantes = {
                    'n_estimators': metadata['parametros'].get('n_estimators'),
                    'max_depth': metadata['parametros'].get('max_depth'),
                    'max_features': metadata['parametros'].get('max_features'),
                    'bootstrap': metadata['parametros'].get('bootstrap'),
                }
                st.json(params_importantes)
                
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
