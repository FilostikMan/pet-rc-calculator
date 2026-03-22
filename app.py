import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Калькулятор RC для ПЭТ",
    page_icon="🔬",
    layout="wide"
)

# --- ДАННЫЕ И МОДЕЛЬ ---
# Коэффициенты полиномиальной регрессии 2-й степени (из предыдущего анализа)
# RC = intercept + c1*d + c2*t + c3*d^2 + c4*d*t + c5*t^2
MODEL_PARAMS = {
    'intercept': -0.3116984512,
    'c_d': 0.0617471346,
    'c_t': 0.0553472204,
    'c_d2': -0.0008712837,
    'c_dt': -0.0005495930,
    'c_t2': -0.0014192538
}

# Исходные данные для графиков валидации
DATA_RAW = """
d,TBR_teor,TBR_prakt,RC
37,6,5.69029069280389,0.993917244
28,6,5.06031666247801,0.86041975
22,6,5.09612968082433,0.86800887
17,6,4.45107648487895,0.731315958
13,6,3.32725978051437,0.4931685
10,6,2.30355198123482,0.276235073
37,8,6.90214124718257,0.904384958
28,8,6.91510142749812,0.906370846
22,8,6.35011269722014,0.819797637
17,8,5.49943651389932,0.689448546
13,8,4.63955672426747,0.557689187
10,8,2.9748309541698,0.302603298
37,10,9.97428346260733,0.998818514
28,10,9.51263013812269,0.960723391
22,10,8.8164171056452,0.882073128
17,10,8.47982081380397,0.844048291
13,10,7.1630926210129,0.695299253
10,10,5.1401966070762,0.466775372
37,14,13.3506952975284,0.9812977
28,14,13.4000786202152,0.985221345
22,14,12.3920200481549,0.905128238
17,14,12.3903002309469,0.904991593
13,14,9.98427595695543,0.713826155
10,14,6.44243526116653,0.432416886
37,20,20.1925843426609,0.974460488
28,20,19.4589076059214,0.937209695
22,20,18.3672096152954,0.881781174
17,20,17.6894519467261,0.847369546
13,20,16.9926678732192,0.811991895
10,20,10.7636085201169,0.495725357
"""

df = pd.read_csv(pd.io.common.StringIO(DATA_RAW))


def calculate_rc(diameter, tbr):
    """Расчет RC по формуле"""
    p = MODEL_PARAMS
    rc = (p['intercept'] +
          p['c_d'] * diameter +
          p['c_t'] * tbr +
          p['c_d2'] * (diameter ** 2) +
          p['c_dt'] * diameter * tbr +
          p['c_t2'] * (tbr ** 2))
    return max(0.0, min(1.0, rc))  # Ограничение 0-1


# --- БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
st.sidebar.header("📥 Параметры очага")
st.sidebar.markdown("Введите параметры для расчета коэффициента восстановления (RC).")

d_input = st.sidebar.slider(
    "Диаметр очага (d), мм",
    min_value=10.0,
    max_value=37.0,
    value=22.0,
    step=0.5
)

tbr_input = st.sidebar.number_input(
    "Практическое TBR (TBR_prakt)",
    min_value=0.0,
    max_value=25.0,
    value=8.0,
    step=0.1
)

# --- ОСНОВНАЯ ЧАСТЬ ---
st.title("🔬 Калькулятор коэффициента восстановления (RC)")
st.markdown("""
Инструмент для коррекции эффекта частичного объёма (PVE) в ПЭТ-визуализации.
Расчёт основан на полиномиальной регрессии 2-й степени по данным фантома NEMA.
""")

# 1. РАСЧЁТ И РЕЗУЛЬТАТ
rc_result = calculate_rc(d_input, tbr_input)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.metric(label="Расчётный RC", value=f"{rc_result:.4f}")

with col2:
    # Визуализация в виде спидометра
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rc_result,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Качество восстановления", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "#ffcccc"},
                {'range': [0.5, 0.8], 'color': "#fff3cd"},
                {'range': [0.8, 1], 'color': "#d4edda"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8}}))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    if rc_result >= 0.8:
        st.success("**Отлично**\nПотери активности минимальны.")
    elif rc_result >= 0.5:
        st.warning("**Средне**\nТребуется коррекция активности.")
    else:
        st.error("**Низкое**\nЗначительные потери сигнала. Коррекция критична.")

st.divider()

# 2. ГРАФИКИ И АНАЛИЗ (ВКЛАДКИ)
tab1, tab2, tab3 = st.tabs(["📊 Графики зависимости", "📈 Валидация модели", "📄 О методе"])

with tab1:
    st.subheader("Зависимость RC от диаметра и TBR")

    # Генерация сетки для поверхности
    d_range = np.linspace(10, 37, 50)
    t_range = np.linspace(2, 20, 50)
    D, T = np.meshgrid(d_range, t_range)
    Z = np.vectorize(calculate_rc)(D, T)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # 3D Поверхность
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=D, y=T, colorscale='Viridis')])
        fig_3d.update_layout(title='3D Поверхность отклика',
                             scene=dict(xaxis_title='Диаметр (мм)', yaxis_title='TBR', zaxis_title='RC'))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_g2:
        # Тепловая карта (Contour)
        fig_contour = go.Figure(data=go.Contour(z=Z, x=d_range, y=t_range, colorscale='Viridis'))
        fig_contour.update_layout(title='Тепловая карта RC', xaxis_title='Диаметр (мм)', yaxis_title='TBR')
        # Добавим точку текущего расчета
        fig_contour.add_trace(go.Scatter(x=[d_input], y=[tbr_input], mode='markers', marker=dict(color='red', size=15),
                                         name='Текущий расчет'))
        st.plotly_chart(fig_contour, use_container_width=True)

with tab2:
    st.subheader("Сравнение фактических и предсказанных значений")

    # Расчет предсказаний для исходных данных
    df['RC_pred'] = df.apply(lambda row: calculate_rc(row['d'], row['TBR_prakt']), axis=1)

    # Scatter plot
    fig_scatter = px.scatter(df, x='RC', y='RC_pred',
                             trendline="ols",
                             labels={'RC': 'Фактический RC', 'RC_pred': 'Предсказанный RC'},
                             title='Фактические vs Предсказанные значения',
                             color='d', color_continuous_scale='Plasma')

    # Линия идеального совпадения
    fig_scatter.add_shape(type="line", line=dict(dash="dash", color="black"),
                          x0=0, y0=0, x1=1, y1=1)

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Таблица ошибок
    df['Error'] = abs(df['RC'] - df['RC_pred'])
    st.write(f"Средняя абсолютная ошибка (MAE): **{df['Error'].mean():.4f}**")
    st.dataframe(df[['d', 'TBR_prakt', 'RC', 'RC_pred', 'Error']].style.format("{:.4f}"))

with tab3:
    st.subheader("Методология расчёта")
    st.markdown("""
    **Математическая модель:**
    Для расчёта коэффициента восстановления (Recovery Coefficient, RC) используется полиномиальная регрессия второго порядка.

    $$ RC = \\beta_0 + \\beta_1 \\cdot d + \\beta_2 \\cdot TBR + \\beta_3 \\cdot d^2 + \\beta_4 \\cdot d \\cdot TBR + \\beta_5 \\cdot TBR^2 $$

    **Коэффициенты модели:**
    - $\\beta_0$ (Intercept): -0.3117
    - $\\beta_1$ (d): 0.0617
    - $\\beta_2$ (TBR): 0.0553
    - $\\beta_3$ (d²): -0.0009
    - $\\beta_4$ (d·TBR): -0.0005
    - $\\beta_5$ (TBR²): -0.0014

    **Ограничения применения:**
    - Диаметр очага: 10 – 37 мм
    - TBR: 2 – 20
    - Экстраполяция за пределы данных не рекомендуется.
    """)

    st.download_button(
        label="📥 Скачать исходные данные (CSV)",
        data=DATA_RAW,
        file_name="nema_phantom_data.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.caption("Разработано для дипломной работы. Модель обучена на данных фантома NEMA IU.")