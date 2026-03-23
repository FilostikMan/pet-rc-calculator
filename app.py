import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Калькулятор коэффициента восстановления для ПЭТ",
    page_icon="🔬",
    layout="wide"
)

# --- ДАННЫЕ И МОДЕЛЬ (ОБНОВЛЕНО: 35 точек) ---
MODEL_PARAMS = {
    'intercept': -0.3461548248969348,
    'c_d':       0.0602579867154475,
    'c_t':       0.0640605623259551,
    'c_d2':      -0.0008076894132690,
    'c_dt':      -0.0006743922635872,
    'c_t2':      -0.0016460658062235
}

# ОБНОВЛЕНО: Добавлены 5 точек при TBR=3 (без сферы 10 мм)
DATA_RAW = """
d,TBR_teor,TBR_prakt,RC
37,3,2.694997839,0.905285942
28,3,2.516012417,0.809691138
22,3,2.333097568,0.711997656
17,3,1.916971197,0.489747606
13,3,1.550945027,0.294255707
37,6,5.690290693,0.993917244
28,6,5.060316662,0.86041975
22,6,5.096129681,0.86800887
17,6,4.451076485,0.731315958
13,6,3.327259781,0.4931685
10,6,2.303551981,0.276235073
37,8,6.902141247,0.904384958
28,8,6.915101427,0.906370846
22,8,6.350112697,0.819797637
17,8,5.499436514,0.689448546
13,8,4.639556724,0.557689187
10,8,2.974830954,0.302603298
37,10,9.974283463,0.998818514
28,10,9.512630138,0.960723391
22,10,8.816417106,0.882073128
17,10,8.479820814,0.844048291
13,10,7.163092621,0.695299253
10,10,5.140196607,0.466775372
37,14,13.3506953,0.9812977
28,14,13.40007862,0.985221345
22,14,12.39202005,0.905128238
17,14,12.39030023,0.904991593
13,14,9.984275957,0.713826155
10,14,6.442435261,0.432416886
37,20,20.19258434,0.974460488
28,20,19.45890761,0.937209695
22,20,18.36720962,0.881781174
17,20,17.68945195,0.847369546
13,20,16.99266787,0.811991895
10,20,10.76360852,0.495725357
"""

df = pd.read_csv(pd.io.common.StringIO(DATA_RAW))


def calculate_rc(diameter, tbr):
    """Расчет RC по формуле полиномиальной регрессии 2-й степени"""
    p = MODEL_PARAMS
    rc = (p['intercept'] +
          p['c_d'] * diameter +
          p['c_t'] * tbr +
          p['c_d2'] * (diameter ** 2) +
          p['c_dt'] * diameter * tbr +
          p['c_t2'] * (tbr ** 2))
    return max(0.0, min(1.0, rc))


# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("📥 Диаметр патологического очага")
st.sidebar.markdown("Введите параметры для расчета коэффициента восстановления (RC).")

d_input = st.sidebar.slider(
    "Диаметр очага (d), мм",
    min_value=10.0,
    max_value=37.0,
    value=22.0,
    step=0.1
)

tbr_input = st.sidebar.number_input(
    "Практическое соотношение объёмных активностей очаг/фон TBR (TBR_prakt)",
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
**Обновлено:** Модель обучена на 35 экспериментальных точках (TBR 3-20).
""")

# 1. РАСЧЁТ И РЕЗУЛЬТАТ
rc_result = calculate_rc(d_input, tbr_input)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.metric(label="Расчётный RC", value=f"{rc_result:.4f}")

with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rc_result,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Коэффициент восстановления", 'font': {'size': 24}},
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
        st.success("**✅ Отлично**\nПотери активности минимальны.")
    elif rc_result >= 0.5:
        st.warning("**⚠️ Средне**\nТребуется коррекция активности.")
    else:
        st.error("**❌ Низкое**\nЗначительные потери сигнала. Коррекция критична.")

st.divider()

# 2. ГРАФИКИ И АНАЛИЗ (ВКЛАДКИ)
tab1, tab2, tab3 = st.tabs(["📊 Графики зависимости", "📈 Валидация модели", "📄 О методе"])

with tab1:
    st.subheader("Зависимость RC от диаметра и TBR")

    d_range = np.linspace(10, 37, 50)
    t_range = np.linspace(2, 20, 50)
    D, T = np.meshgrid(d_range, t_range)
    Z = np.vectorize(calculate_rc)(D, T)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=D, y=T, colorscale='Viridis')])
        fig_3d.update_layout(title='3D Поверхность отклика',
                             scene=dict(xaxis_title='Диаметр (мм)', yaxis_title='TBR', zaxis_title='RC'))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_g2:
        fig_contour = go.Figure(data=go.Contour(z=Z, x=d_range, y=t_range, colorscale='Viridis'))
        fig_contour.update_layout(title='Тепловая карта RC', xaxis_title='Диаметр (мм)', yaxis_title='TBR')
        fig_contour.add_trace(go.Scatter(x=[d_input], y=[tbr_input], mode='markers', marker=dict(color='red', size=15),
                                         name='Текущий расчет'))
        st.plotly_chart(fig_contour, use_container_width=True)

with tab2:
    st.subheader("Сравнение фактических и предсказанных значений")

    df['RC_pred'] = df.apply(lambda row: calculate_rc(row['d'], row['TBR_prakt']), axis=1)

    # === РАСЧЁТ МЕТРИК КАЧЕСТВА МОДЕЛИ ===
    df['Error'] = abs(df['RC'] - df['RC_pred'])
    df['Rel_Error'] = (df['Error'] / df['RC']) * 100

    # Коэффициент корреляции Пирсона (R)
    r_value, p_value = stats.pearsonr(df['RC'], df['RC_pred'])

    # Коэффициент детерминации (R²)
    ss_res = ((df['RC'] - df['RC_pred']) ** 2).sum()
    ss_tot = ((df['RC'] - df['RC'].mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    # Линия тренда (вручную, без statsmodels)
    x_min, x_max = df['RC'].min(), df['RC'].max()
    x_line = np.linspace(x_min, x_max, 100)
    slope, intercept, _, _, _ = stats.linregress(df['RC'], df['RC_pred'])
    y_line = slope * x_line + intercept

    # Остальные метрики
    mae = df['Error'].mean()
    mre = df['Rel_Error'].mean()
    rmse = np.sqrt(((df['RC'] - df['RC_pred']) ** 2).mean())

    # === ГРАФИК ===
    fig_scatter = px.scatter(df, x='RC', y='RC_pred',
                             labels={'RC': 'Фактический RC', 'RC_pred': 'Предсказанный RC'},
                             title='Фактические vs Предсказанные значения',
                             color='d',
                             color_discrete_sequence=px.colors.sequential.Plasma)

    # Линия идеального совпадения (y=x)
    fig_scatter.add_shape(type="line", line=dict(dash="dash", color="black", width=2),
                          x0=x_min, y0=x_min, x1=x_max, y1=x_max,
                          name='Идеальное совпадение')

    # Линия регрессии
    fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                     line=dict(color='red', dash='dot', width=2),
                                     name=f'Линия тренда (R²={r_squared:.4f})'))

    st.plotly_chart(fig_scatter, use_container_width=True)

    # === МЕТРИКИ КАЧЕСТВА МОДЕЛИ ===
    st.markdown("### 📊 Метрики качества модели")

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric("Коэффициент корреляции (R)", f"{r_value:.4f}")

    with col_m2:
        st.metric("Коэффициент детерминации (R²)", f"{r_squared:.4f}")

    with col_m3:
        st.metric("Средняя абсолютная ошибка (MAE)", f"{mae:.4f}")

    with col_m4:
        st.metric("Средняя относит. погрешность (MRE)", f"{mre:.2f}%")

    with col_m5:
        st.metric("Среднеквадратичная ошибка (RMSE)", f"{rmse:.4f}")

    # === ИНТЕРПРЕТАЦИЯ МЕТРИК ===
    st.markdown("### 📋 Интерпретация метрик")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.info("**Коэффициент корреляции Пирсона (R):**")
        st.write(f"Значение: **{r_value:.4f}**")
        if r_value >= 0.9:
            st.success("✅ Очень сильная связь между фактическими и предсказанными значениями")
        elif r_value >= 0.7:
            st.warning("⚠️ Сильная связь")
        else:
            st.error("❌ Средняя или слабая связь")

    with col_i2:
        st.info("**Коэффициент детерминации (R²):**")
        st.write(f"Значение: **{r_squared:.4f}**")
        st.write(f"Модель объясняет **{r_squared * 100:.1f}%** дисперсии данных")
        if r_squared >= 0.8:
            st.success("✅ Отличная точность модели")
        elif r_squared >= 0.6:
            st.warning("⚠️ Хорошая точность модели")
        else:
            st.error("❌ Требуется улучшение модели")

    st.markdown("### 📋 Таблица погрешностей по экспериментам")

    styled_df = df[['d', 'TBR_prakt', 'RC', 'RC_pred', 'Error', 'Rel_Error']].copy()
    styled_df.columns = ['d (мм)', 'TBR_prakt', 'RC факт', 'RC предск', 'Δ абсолютная', 'Δ относительная (%)']

    st.dataframe(
        styled_df.style.format({
            'd (мм)': '{:.0f}',
            'TBR_prakt': '{:.1f}',
            'RC факт': '{:.4f}',
            'RC предск': '{:.4f}',
            'Δ абсолютная': '{:.4f}',
            'Δ относительная (%)': '{:.2f}%'
        }),
        use_container_width=True
    )

    # Статистика погрешностей
    st.markdown("### 📊 Статистика погрешностей")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.write("**Абсолютная погрешность:**")
        st.write(f"- Мин: {df['Error'].min():.4f}")
        st.write(f"- Макс: {df['Error'].max():.4f}")
        st.write(f"- Среднее: {df['Error'].mean():.4f}")

    with col_s2:
        st.write("**Относительная погрешность:**")
        st.write(f"- Мин: {df['Rel_Error'].min():.2f}%")
        st.write(f"- Макс: {df['Rel_Error'].max():.2f}%")
        st.write(f"- Среднее: {df['Rel_Error'].mean():.2f}%")

with tab3:
    st.subheader("Методология расчёта")
    st.markdown(f"""
    **Математическая модель:**
    Для расчёта коэффициента восстановления (Recovery Coefficient, RC) используется полиномиальная регрессия второго порядка.

    $$ RC = \\beta_0 + \\beta_1 \\cdot d + \\beta_2 \\cdot TBR + \\beta_3 \\cdot d^2 + \\beta_4 \\cdot d \\cdot TBR + \\beta_5 \\cdot TBR^2 $$

    **Коэффициенты модели (обновлено, n=35):**
    - $\\beta_0$ (Intercept): {MODEL_PARAMS['intercept']:.4f}
    - $\\beta_1$ (d): {MODEL_PARAMS['c_d']:.4f}
    - $\\beta_2$ (TBR): {MODEL_PARAMS['c_t']:.4f}
    - $\\beta_3$ (d²): {MODEL_PARAMS['c_d2']:.4f}
    - $\\beta_4$ (d·TBR): {MODEL_PARAMS['c_dt']:.4f}
    - $\\beta_5$ (TBR²): {MODEL_PARAMS['c_t2']:.4f}

    **Характеристики выборки:**
    - Количество наблюдений: **35**
    - Диапазон диаметров: **10-37 мм** (6 сфер фантома NEMA)
    - Диапазон TBR: **1.5-20** (6 уровней: 3, 6, 8, 10, 14, 20)
    - При TBR=3 сфера 10 мм не измерялась (низкий сигнал/шум)

    **Ограничения применения:**
    - Диаметр очага: 10 – 37 мм
    - TBR: 1.5 – 20
    - Экстраполяция за пределы данных не рекомендуется.

    **Метрики качества модели:**
    | Метрика | Описание | Оценка |
    |---------|----------|--------|
    | **R (Пирсон)** | Коэффициент корреляции | 0.95+ = отлично |
    | **R²** | Доля объяснённой дисперсии | 0.80+ = отлично |
    | **MAE** | Средняя абсолютная ошибка | Чем меньше, тем лучше |
    | **MRE** | Средняя относительная погрешность | <10% = отлично |
    | **RMSE** | Среднеквадратичная ошибка | Чувствительна к выбросам |
    """)

    st.download_button(
        label="📥 Скачать исходные данные (CSV)",
        data=DATA_RAW,
        file_name="nema_phantom_data_35points.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Разработано для дипломной работы. Модель обучена на данных фантома NEMA IU (35 точек, 2026).")