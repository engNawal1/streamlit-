# -*- coding: utf-8 -*-
"""
قسم أمراض القلب — لوحة تحليل وتقييم مخاطر القلب المبكر (EarlyMed)
تطبيق Streamlit تفاعلي مبني على مشروع نوال لتصنيف مخاطر أمراض القلب
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="قسم أمراض القلب | EarlyMed",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# نظام الألوان و CSS (Pastel Clinical Theme)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
    --bg:            #F6F8FC;
    --bg-card:       #FFFFFF;
    --teal:          #8FD3C9;
    --teal-deep:     #4FA69A;
    --lavender:      #C9CBF2;
    --blush:         #F6C6C7;
    --blush-deep:    #E8898C;
    --amber:         #FBE0A8;
    --amber-deep:    #E3A93F;
    --ink:           #2C3242;
    --ink-soft:      #6B7288;
    --line:          #E7EAF3;
    --ok:            #79C6A6;
    --warn:          #E8B65B;
    --danger:        #E37E82;
}

html, body, [class*="css"]  {
    font-family: 'Tajawal', sans-serif !important;
    direction: rtl;
}

.stApp{
    background:
        radial-gradient(circle at 8% 0%, rgba(143,211,201,0.16), transparent 42%),
        radial-gradient(circle at 96% 12%, rgba(201,203,242,0.20), transparent 40%),
        var(--bg);
}

/* إخفاء عناصر Streamlit الافتراضية */
#MainMenu, footer, header {visibility:hidden;}
.block-container{padding-top:1.6rem; max-width:1250px;}

/* ===== الشريط الجانبي ===== */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #EFF3FC 0%, #F6F8FC 55%);
    border-left: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container{padding-top:2rem;}

.brand{
    display:flex; align-items:center; gap:.6rem;
    padding: .4rem .2rem 1.2rem .2rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.1rem;
}
.brand-badge{
    width:44px; height:44px; border-radius:14px;
    background: linear-gradient(135deg, var(--teal), var(--lavender));
    display:flex; align-items:center; justify-content:center;
    font-size:22px; box-shadow: 0 6px 14px rgba(79,166,154,0.28);
}
.brand-title{font-weight:900; font-size:1.05rem; color:var(--ink); line-height:1.15;}
.brand-sub{font-size:.72rem; color:var(--ink-soft); letter-spacing:.3px;}

/* ===== الهيدر العلوي ===== */
.hero{
    background: linear-gradient(120deg, #FFFFFF 0%, #F1F6FB 100%);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.7rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero:before{
    content:"";
    position:absolute; inset:0;
    background: radial-gradient(circle at 100% 0%, rgba(246,198,199,0.35), transparent 45%);
}
.hero-eyebrow{
    display:inline-flex; align-items:center; gap:.4rem;
    background: rgba(143,211,201,0.22); color: var(--teal-deep);
    font-size:.72rem; font-weight:700; padding:.28rem .75rem;
    border-radius:999px; margin-bottom:.7rem;
}
.hero h1{font-size:1.65rem; font-weight:900; color:var(--ink); margin:0 0 .35rem 0;}
.hero p{color:var(--ink-soft); font-size:.92rem; margin:0; max-width:640px; line-height:1.65;}

/* ===== نبضة ECG ===== */
.ecg-wrap{position:relative; z-index:1;}
.ecg-line path{
    stroke: var(--blush-deep);
    stroke-width: 2.4;
    fill:none;
    stroke-dasharray: 400;
    stroke-dashoffset: 400;
    animation: draw 3.2s ease-in-out infinite;
}
@keyframes draw{
    0%{stroke-dashoffset:400;}
    55%{stroke-dashoffset:0;}
    100%{stroke-dashoffset:-400;}
}

/* ===== بطاقات المقاييس ===== */
.kpi{
    background: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    transition: transform .18s ease, box-shadow .18s ease;
    height:100%;
}
.kpi:hover{ transform: translateY(-3px); box-shadow: 0 14px 26px rgba(44,50,66,0.08);}
.kpi .icon{font-size:1.5rem; margin-bottom:.35rem; display:block;}
.kpi .val{font-size:1.55rem; font-weight:900; color:var(--ink);}
.kpi .lbl{font-size:.78rem; color:var(--ink-soft); font-weight:500;}
.kpi .delta{font-size:.72rem; font-weight:700; margin-top:.25rem; display:inline-block; padding:.1rem .5rem; border-radius:999px;}
.d-ok{background:rgba(121,198,166,.18); color:#2E8A63;}
.d-warn{background:rgba(232,182,91,.2); color:#B27A16;}
.d-info{background:rgba(201,203,242,.28); color:#5457A6;}

/* ===== بطاقة قسم ===== */
.section-card{
    background: var(--bg-card);
    border:1px solid var(--line);
    border-radius:20px;
    padding: 1.5rem 1.6rem;
    margin-bottom:1.2rem;
}
.section-card h3{
    font-size:1.05rem; font-weight:800; color:var(--ink); margin:0 0 .2rem 0;
    display:flex; align-items:center; gap:.5rem;
}
.section-card .hint{color:var(--ink-soft); font-size:.82rem; margin-bottom:1rem;}

/* ===== تبويبات ===== */
.stTabs [data-baseweb="tab-list"]{gap:6px; background:transparent;}
.stTabs [data-baseweb="tab"]{
    background: #EEF2FA; border-radius:12px; padding:.55rem 1.1rem;
    font-weight:700; color:var(--ink-soft); border:1px solid transparent;
}
.stTabs [aria-selected="true"]{
    background: linear-gradient(135deg, var(--teal), #B7E4DB) !important;
    color:#1D4E47 !important; border:1px solid var(--teal-deep);
}

/* ===== أزرار الراديو داخل الشريط الجانبي كقائمة تنقل ===== */
div[role="radiogroup"] > label{
    background:#fff !important; border:1px solid var(--line); border-radius:12px;
    padding:.55rem .8rem; margin-bottom:.45rem; width:100%; font-weight:600;
    color: var(--ink-soft) !important; transition:.15s;
}
div[role="radiogroup"] > label:hover{border-color:var(--teal-deep); color:var(--ink) !important;}
div[role="radiogroup"] > label *{ color: inherit !important; }
div[role="radiogroup"] > label p,
div[role="radiogroup"] > label span,
div[role="radiogroup"] > label div{ color: var(--ink-soft) !important; }

/* ===== تثبيت ألوان النصوص الافتراضية بغض النظر عن وضع النظام (Dark/Light) ===== */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div,
section[data-testid="stSidebar"] *{
    color: var(--ink);
}
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] .stMarkdown *{
    color: var(--ink-soft) !important;
}
.stRadio label span p{ color: inherit !important; }
[data-testid="stWidgetLabel"] p{ color: var(--ink) !important; }
[data-testid="stMarkdownContainer"] p{ color: inherit; }

/* ===== زر رئيسي ===== */
.stButton>button{
    background: linear-gradient(135deg, var(--teal-deep), var(--teal));
    color:#fff; border:none; border-radius:14px; padding:.65rem 1.6rem;
    font-weight:800; font-size:.95rem; box-shadow: 0 10px 20px rgba(79,166,154,0.28);
    transition: transform .15s ease;
}
.stButton>button:hover{ transform: translateY(-2px); }

/* ===== بطاقة نتيجة الخطر ===== */
.risk-card{
    border-radius:22px; padding:1.8rem; text-align:center;
    border:1px solid var(--line);
}
.risk-low{ background: linear-gradient(135deg, #E8F7F1, #F6FBF9);}
.risk-mid{ background: linear-gradient(135deg, #FDF3DF, #FEFAF0);}
.risk-high{ background: linear-gradient(135deg, #FCE9EA, #FFF6F6);}
.risk-title{font-size:1.1rem; font-weight:800; color:var(--ink);}
.risk-badge{
    display:inline-block; margin-top:.6rem; padding:.4rem 1.1rem;
    border-radius:999px; font-weight:900; font-size:1.4rem;
}

.badge-low{ background:var(--ok); color:#fff;}
.badge-mid{ background:var(--warn); color:#fff;}
.badge-high{ background:var(--danger); color:#fff;}

footer-note{color:var(--ink-soft);}
hr{border-color:var(--line);}

/* شارات الأعراض */
.symptom-pill{
    display:inline-flex; align-items:center; gap:.4rem;
    background:#F0F3FA; border:1px solid var(--line); border-radius:999px;
    padding:.3rem .8rem; font-size:.78rem; font-weight:600; color:var(--ink-soft); margin:2px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SYMPTOM_COLS = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol",
]

SYMPTOM_LABELS_AR = {
    "Chest_Pain": "ألم في الصدر",
    "Shortness_of_Breath": "ضيق التنفس",
    "Fatigue": "الإرهاق العام",
    "Palpitations": "خفقان القلب",
    "Dizziness": "الدوخة",
    "Swelling": "تورّم الأطراف",
    "Pain_Arms_Jaw_Back": "ألم بالذراع/الفك/الظهر",
    "Cold_Sweats_Nausea": "تعرق بارد وغثيان",
    "High_BP": "ارتفاع ضغط الدم",
    "High_Cholesterol": "ارتفاع الكوليسترول",
}

SYMPTOM_ICONS = {
    "Chest_Pain": "🫀", "Shortness_of_Breath": "💨", "Fatigue": "🥱",
    "Palpitations": "💓", "Dizziness": "😵‍💫", "Swelling": "🦵",
    "Pain_Arms_Jaw_Back": "💪", "Cold_Sweats_Nausea": "🥶",
    "High_BP": "🩺", "High_Cholesterol": "🧬",
}

# ============================================================
# تحميل البيانات
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    """
    يحاول تحميل ملف بيانات EarlyMed من عدة مسارات شائعة.
    إن لم يجده، يولّد بيانات اصطناعية بنفس البنية والتوزيع
    (كل عرض ~Bernoulli(0.5)) حتى تعمل اللوحة مباشرة للعرض التجريبي.
    """
    candidate_paths = [
        "heart_disease_risk_dataset_earlymed-selected-columns.csv",
        "/content/heart_disease_risk_dataset_earlymed-selected-columns.csv",
        "/mnt/user-data/uploads/heart_disease_risk_dataset_earlymed-selected-columns.csv",
    ]
    for p in candidate_paths:
        try:
            df = pd.read_csv(p)
            return df, True
        except Exception:
            continue

    rng = np.random.default_rng(42)
    n = 70000
    data = {c: rng.binomial(1, 0.5, n).astype(float) for c in SYMPTOM_COLS}
    df = pd.DataFrame(data)
    return df, False


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    heart = df.copy()
    heart["Total_Symptoms"] = heart[SYMPTOM_COLS].sum(axis=1)
    heart["Heart_Risk"] = (heart["Total_Symptoms"] >= 4).astype(int)

    X = heart.drop(["Total_Symptoms", "Heart_Risk"], axis=1)
    y = heart["Heart_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:, 1]

    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)
    proba_log = log_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf),
            "Recall": recall_score(y_test, y_pred_rf),
            "F1": f1_score(y_test, y_pred_rf),
            "ROC-AUC": roc_auc_score(y_test, proba_rf),
            "y_pred": y_pred_rf, "proba": proba_rf,
        },
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, y_pred_log),
            "Precision": precision_score(y_test, y_pred_log),
            "Recall": recall_score(y_test, y_pred_log),
            "F1": f1_score(y_test, y_pred_log),
            "ROC-AUC": roc_auc_score(y_test, proba_log),
            "y_pred": y_pred_log, "proba": proba_log,
        },
    }

    return {
        "heart": heart, "X": X, "y": y,
        "X_test": X_test, "y_test": y_test,
        "rf": rf, "log_model": log_model, "scaler": scaler,
        "metrics": metrics,
    }


df_raw, is_real_data = load_data()
bundle = train_models(df_raw)
heart = bundle["heart"]

# ============================================================
# الشريط الجانبي — التنقل
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-badge">💗</div>
            <div>
                <div class="brand-title">قسم أمراض القلب</div>
                <div class="brand-sub">مبادرة EarlyMed للكشف المبكر</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "التنقل",
        ["🏠  نظرة عامة", "📊  استكشاف البيانات", "🩺  حاسبة الخطر", "⚖️  مقارنة النماذج", "ℹ️  حول المشروع"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:.78rem; color:var(--ink-soft); line-height:1.9;">
        📁 عدد الحالات: <b>{len(heart):,}</b><br>
        🧪 مصدر البيانات: <b>{"ملف حقيقي" if is_real_data else "بيانات تجريبية"}</b><br>
        🕒 آخر تحديث للنموذج: عند تشغيل التطبيق
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_real_data:
        st.info(
            "لم يتم العثور على ملف EarlyMed الأصلي، فتم توليد بيانات تجريبية "
            "بنفس البنية لعرض اللوحة. ضعي الملف بجانب app.py لعرض بياناتك الحقيقية.",
            icon="💡",
        )

# ============================================================
# رأس الصفحة العام
# ============================================================
def hero(title, subtitle, eyebrow):
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-eyebrow">🫀 {eyebrow}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="ecg-wrap">
                <svg class="ecg-line" width="100%" height="34" viewBox="0 0 600 34" preserveAspectRatio="none">
                    <path d="M0,17 L120,17 L140,4 L160,30 L180,17 L230,17 L245,10 L260,24 L275,17 L600,17" />
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon, value, label, delta=None, delta_type="ok"):
    delta_html = f'<div class="delta d-{delta_type}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi">
            <span class="icon">{icon}</span>
            <div class="val">{value}</div>
            <div class="lbl">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


PASTEL_SEQ = ["#8FD3C9", "#C9CBF2", "#F6C6C7", "#FBE0A8", "#A9D3F5", "#D9C7F0"]

def style_fig(fig, height=380):
    fig.update_layout(
        template="simple_white",
        font=dict(family="Tajawal, sans-serif", color="#2C3242"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10),
        height=height,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig

# ============================================================
# الصفحة 1 — نظرة عامة
# ============================================================
if page.startswith("🏠"):
    hero(
        "لوحة تقييم مخاطر القلب المبكرة",
        "نظرة شاملة وتفاعلية على بيانات المرضى، الأعراض الأكثر شيوعًا، ونسب الخطورة "
        "المستخلَصة من نموذج EarlyMed — لدعم فرق الرعاية في اتخاذ قرارات مبكرة وواعية.",
        "لوحة القسم",
    )

    risk_rate = heart["Heart_Risk"].mean() * 100
    avg_sym = heart["Total_Symptoms"].mean()
    top_symptom = heart[SYMPTOM_COLS].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("🧑‍🤝‍🧑", f"{len(heart):,}", "إجمالي عدد الحالات", "قاعدة بيانات EarlyMed", "info")
    with c2:
        kpi_card("⚠️", f"{risk_rate:.1f}%", "نسبة الحالات عالية الخطورة",
                  "خطورة إن ≥ 4 أعراض", "warn" if risk_rate > 40 else "ok")
    with c3:
        kpi_card("📈", f"{avg_sym:.1f}", "متوسط عدد الأعراض لكل مريض", "من أصل 10 أعراض", "info")
    with c4:
        kpi_card(SYMPTOM_ICONS.get(top_symptom, "🩺"), SYMPTOM_LABELS_AR.get(top_symptom, top_symptom),
                  "العرَض الأكثر شيوعًا", "الأعلى انتشارًا في العينة", "ok")

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([1.35, 1])

    with colA:
        st.markdown('<div class="section-card"><h3>📊 انتشار الأعراض بين المرضى</h3>'
                     '<div class="hint">نسبة المرضى الذين أبلغوا عن كل عرض</div></div>', unsafe_allow_html=True)
        prevalence = (heart[SYMPTOM_COLS].mean() * 100).sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=prevalence.values,
            y=[SYMPTOM_LABELS_AR[c] for c in prevalence.index],
            orientation="h",
            marker=dict(color=PASTEL_SEQ * 3, line=dict(color="#E7EAF3", width=1)),
            text=[f"{v:.1f}%" for v in prevalence.values],
            textposition="outside",
        ))
        fig.update_xaxes(title="نسبة المرضى (%)")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with colB:
        st.markdown('<div class="section-card"><h3>💗 توزيع مستوى الخطورة</h3>'
                     '<div class="hint">منخفض مقابل مرتفع الخطورة</div></div>', unsafe_allow_html=True)
        risk_counts = heart["Heart_Risk"].value_counts().rename({0: "خطورة منخفضة", 1: "خطورة مرتفعة"})
        fig = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=.62,
            marker=dict(colors=["#8FD3C9", "#F6C6C7"]),
            textinfo="percent",
        ))
        fig.update_layout(annotations=[dict(text="الخطورة", showarrow=False, font_size=14)])
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown('<div class="section-card"><h3>📉 توزيع إجمالي عدد الأعراض لكل مريض</h3>'
                 '<div class="hint">خط أحمر يشير إلى عتبة التصنيف كـ"خطورة مرتفعة" (٤ أعراض فأكثر)</div></div>',
                 unsafe_allow_html=True)
    fig = px.histogram(heart, x="Total_Symptoms", nbins=11, color_discrete_sequence=["#8FD3C9"])
    fig.add_vline(x=4, line_dash="dash", line_color="#E37E82",
                  annotation_text="عتبة الخطورة", annotation_position="top")
    fig.update_xaxes(title="إجمالي عدد الأعراض")
    fig.update_yaxes(title="عدد المرضى")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

# ============================================================
# الصفحة 2 — استكشاف البيانات
# ============================================================
elif page.startswith("📊"):
    hero(
        "استكشاف البيانات التفاعلي",
        "افحصي العلاقات بين الأعراض، رشّحي حسب الأعراض المختارة، وتعرّفي على الأنماط "
        "الكامنة في بيانات المرضى بشكل تفاعلي بالكامل.",
        "EDA",
    )

    tab1, tab2, tab3 = st.tabs(["🔗 خريطة الارتباط", "🎚️ الفلترة التفاعلية", "🧮 جدول البيانات"])

    with tab1:
        st.markdown('<div class="section-card"><h3>خريطة الارتباط الحرارية بين الأعراض</h3>'
                     '<div class="hint">كلما اقترب اللون من الوردي/الأحمر، زادت العلاقة الإيجابية بين عرضين</div></div>',
                     unsafe_allow_html=True)
        corr = heart[SYMPTOM_COLS + ["Total_Symptoms", "Heart_Risk"]].corr()
        labels_full = [SYMPTOM_LABELS_AR.get(c, c) for c in corr.columns[:-2]] + ["إجمالي الأعراض", "الخطورة"]
        pastel_scale = [[0, "#8FD3C9"], [0.5, "#FFFFFF"], [1, "#F6A3A6"]]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=labels_full, y=labels_full,
            colorscale=pastel_scale, zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-card"><h3>فلترة الحالات حسب الأعراض</h3>'
                     '<div class="hint">اختاري أعراضًا لعرض عدد الحالات المطابقة ونسبة خطورتها</div></div>',
                     unsafe_allow_html=True)
        chosen = st.multiselect(
            "اختاري الأعراض الحالية لدى المريض",
            options=SYMPTOM_COLS,
            format_func=lambda c: f"{SYMPTOM_ICONS.get(c,'')} {SYMPTOM_LABELS_AR.get(c,c)}",
        )
        if chosen:
            mask = (heart[chosen] == 1).all(axis=1)
            filtered = heart[mask]
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi_card("🔎", f"{len(filtered):,}", "عدد الحالات المطابقة", None)
            with c2:
                rr = filtered["Heart_Risk"].mean() * 100 if len(filtered) else 0
                kpi_card("⚠️", f"{rr:.1f}%", "نسبة الخطورة المرتفعة ضمن الفئة", None)
            with c3:
                base = heart["Heart_Risk"].mean() * 100
                diff = rr - base
                kpi_card("📊", f"{diff:+.1f} نقطة", "الفرق عن المتوسط العام", None)
        else:
            st.markdown(
                '<div style="color:var(--ink-soft); padding:.6rem 0;">اختاري عرضًا واحدًا على الأقل لعرض النتائج ↑</div>',
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown('<div class="section-card"><h3>عيّنة من بيانات المرضى</h3></div>', unsafe_allow_html=True)
        display_df = heart.rename(columns=SYMPTOM_LABELS_AR).rename(
            columns={"Total_Symptoms": "إجمالي الأعراض", "Heart_Risk": "خطورة مرتفعة"}
        )
        st.dataframe(display_df.head(200), use_container_width=True, height=420)

# ============================================================
# الصفحة 3 — حاسبة الخطر (تفاعلية)
# ============================================================
elif page.startswith("🩺"):
    hero(
        "حاسبة تقييم خطر القلب",
        "أداة دعم قرار سريعة: فعّلي الأعراض التي يعاني منها المريض، وستقوم اللوحة "
        "بحساب احتمالية الخطورة باستخدام نموذجَي Random Forest و Logistic Regression فورًا.",
        "أداة تفاعلية",
    )

    colForm, colResult = st.columns([1.15, 1])

    with colForm:
        st.markdown('<div class="section-card"><h3>📋 نموذج تقييم المريض</h3>'
                     '<div class="hint">فعّلي كل عرض ينطبق على حالة المريض الحالية</div></div>',
                     unsafe_allow_html=True)

        inputs = {}
        cols = st.columns(2)
        for i, sym in enumerate(SYMPTOM_COLS):
            with cols[i % 2]:
                inputs[sym] = st.toggle(
                    f"{SYMPTOM_ICONS.get(sym,'')}  {SYMPTOM_LABELS_AR.get(sym, sym)}",
                    value=False, key=f"tog_{sym}",
                )

        model_choice = st.radio(
            "اختاري النموذج المستخدم للتنبؤ",
            ["Random Forest", "Logistic Regression"],
            horizontal=True,
        )
        predict_clicked = st.button("🔍 احسبي درجة الخطورة", use_container_width=True)

    with colResult:
        active = [s for s, v in inputs.items() if v]
        n_active = len(active)

        if predict_clicked or n_active > 0:
            x_input = pd.DataFrame([[1 if inputs[c] else 0 for c in SYMPTOM_COLS]], columns=SYMPTOM_COLS)

            if model_choice == "Random Forest":
                proba = bundle["rf"].predict_proba(x_input)[0, 1]
            else:
                x_scaled = bundle["scaler"].transform(x_input)
                proba = bundle["log_model"].predict_proba(x_scaled)[0, 1]

            pct = proba * 100
            if pct < 33:
                level, css_card, css_badge = "خطورة منخفضة", "risk-low", "badge-low"
                advice = "لا توجد مؤشرات مقلقة حاليًا. يُنصح بالمتابعة الدورية الروتينية."
            elif pct < 66:
                level, css_card, css_badge = "خطورة متوسطة", "risk-mid", "badge-mid"
                advice = "يُنصح بإجراء فحوصات إضافية واستشارة طبيب القلب في أقرب وقت مناسب."
            else:
                level, css_card, css_badge = "خطورة مرتفعة", "risk-high", "badge-high"
                advice = "يُنصح بتحويل الحالة فورًا لطبيب القلب المختص لإجراء تقييم عاجل."

            st.markdown(
                f"""
                <div class="risk-card {css_card}">
                    <div class="risk-title">نتيجة التقييم — {model_choice}</div>
                    <div class="risk-badge {css_badge}">{pct:.0f}%</div>
                    <div style="font-weight:800; color:var(--ink); margin-top:.4rem;">{level}</div>
                    <div style="color:var(--ink-soft); font-size:.85rem; margin-top:.6rem; line-height:1.7;">{advice}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6B7288"},
                    "bar": {"color": "#4FA69A"},
                    "steps": [
                        {"range": [0, 33], "color": "#E4F5EF"},
                        {"range": [33, 66], "color": "#FDF0D8"},
                        {"range": [66, 100], "color": "#FBE1E2"},
                    ],
                    "threshold": {"line": {"color": "#E37E82", "width": 3}, "value": pct},
                },
            ))
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)

            st.markdown(f'<div class="hint">عدد الأعراض المُفعّلة: <b>{n_active} / 10</b></div>', unsafe_allow_html=True)
            pills = "".join(
                f'<span class="symptom-pill">{SYMPTOM_ICONS.get(s,"")} {SYMPTOM_LABELS_AR.get(s,s)}</span>'
                for s in active
            )
            if pills:
                st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="section-card" style="text-align:center; color:var(--ink-soft);">
                    <div style="font-size:2.2rem;">🩺</div>
                    فعّلي أعراض المريض من النموذج المجاور لعرض نتيجة تقييم الخطورة هنا
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="hint" style="margin-top:1rem;">⚠️ هذه الأداة لأغراض تعليمية وتدعم اتخاذ القرار فقط، '
        'ولا تُغني عن التقييم السريري المتخصص.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# الصفحة 4 — مقارنة النماذج
# ============================================================
elif page.startswith("⚖️"):
    hero(
        "مقارنة أداء النماذج",
        "مقارنة تفصيلية بين نموذجَي Random Forest و Logistic Regression على بيانات "
        "الاختبار: الدقة، الاستدعاء، ومنحنى ROC.",
        "تقييم النماذج",
    )

    m = bundle["metrics"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metric_ar = {"Accuracy": "الدقة", "Precision": "الدقة الموجبة", "Recall": "الاستدعاء",
                 "F1": "F1 Score", "ROC-AUC": "ROC-AUC"}

    c1, c2 = st.columns(2)
    for col, name, icon in zip([c1, c2], ["Random Forest", "Logistic Regression"], ["🌲", "📈"]):
        with col:
            st.markdown(
                f'<div class="section-card"><h3>{icon} {name}</h3></div>', unsafe_allow_html=True
            )
            for mn in metric_names:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                background:#F6F8FC; border-radius:10px; padding:.5rem .9rem; margin-bottom:.4rem;">
                        <span style="font-weight:600; color:var(--ink-soft); font-size:.85rem;">{metric_ar[mn]}</span>
                        <span style="font-weight:900; color:var(--teal-deep); font-family:'IBM Plex Mono';">
                            {m[name][mn]*100:.1f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="section-card"><h3>📊 مقارنة المقاييس جنبًا إلى جنب</h3></div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "المقياس": [metric_ar[x] for x in metric_names] * 2,
            "القيمة": [m["Random Forest"][x] * 100 for x in metric_names] + [m["Logistic Regression"][x] * 100 for x in metric_names],
            "النموذج": ["Random Forest"] * 5 + ["Logistic Regression"] * 5,
        })
        fig = px.bar(comp_df, x="المقياس", y="القيمة", color="النموذج", barmode="group",
                     color_discrete_map={"Random Forest": "#8FD3C9", "Logistic Regression": "#C9CBF2"})
        fig.update_yaxes(title="النسبة (%)")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with colB:
        st.markdown('<div class="section-card"><h3>🎯 منحنى ROC</h3></div>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, color in [("Random Forest", "#4FA69A"), ("Logistic Regression", "#8486D6")]:
            fpr, tpr, _ = roc_curve(bundle["y_test"], m[name]["proba"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f'{name} (AUC={m[name]["ROC-AUC"]:.3f})',
                                      line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="عشوائي",
                                  line=dict(color="#C7CBDA", dash="dash")))
        fig.update_xaxes(title="معدل الإيجابية الخاطئة")
        fig.update_yaxes(title="معدل الإيجابية الصحيحة")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.markdown('<div class="section-card"><h3>🧮 مصفوفات الالتباس</h3></div>', unsafe_allow_html=True)
    colC, colD = st.columns(2)
    for col, name, scale in [(colC, "Random Forest", [[0, "#F6F8FC"], [1, "#4FA69A"]]),
                              (colD, "Logistic Regression", [[0, "#F6F8FC"], [1, "#8486D6"]])]:
        with col:
            cm = confusion_matrix(bundle["y_test"], m[name]["y_pred"])
            fig = go.Figure(go.Heatmap(
                z=cm, x=["متوقع: منخفض", "متوقع: مرتفع"], y=["فعلي: منخفض", "فعلي: مرتفع"],
                colorscale=scale, text=cm, texttemplate="%{text}", textfont=dict(size=16, color="#2C3242"),
                showscale=False,
            ))
            fig.update_layout(title=name)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

# ============================================================
# الصفحة 5 — حول المشروع
# ============================================================
else:
    hero(
        "حول المشروع",
        "ملخص منهجية العمل على بيانات EarlyMed لتصنيف مخاطر أمراض القلب المبكرة.",
        "توثيق",
    )
    st.markdown(
        """
        <div class="section-card">
            <h3>🧭 ملخص المنهجية</h3>
            <p style="color:var(--ink-soft); line-height:2;">
            تم بناء خط أنابيب تعلم آلي كامل للتنبؤ بمخاطر القلب باستخدام بيانات
            EarlyMed (نحو 70,000 حالة). البيانات نظيفة ولا تحتوي على قيم مفقودة،
            وجميع الأعمدة ثنائية (0/1) بحيث لم تكن هناك حاجة لأي ترميز إضافي.<br><br>
            تمّت إضافة عمودين مساعدين: <b>Total_Symptoms</b> (إجمالي عدد الأعراض لكل مريض)
            و<b>Heart_Risk</b> (تصنيف الخطورة عند وجود 4 أعراض فأكثر). تم تطبيق
            <b>MinMaxScaler</b> قبل نموذج الانحدار اللوجستي نظرًا لحساسيته لمقياس الميزات.<br><br>
            تمت مقارنة نموذجين: <b>Random Forest</b> الذي حقق أداءً شبه مثالي،
            و<b>Logistic Regression</b> الذي حقق دقة جيدة (~84%) لكن أقل من الغابة العشوائية.
            </p>
        </div>
        <div class="section-card">
            <h3>🔮 خطوات مقترحة للتطوير</h3>
            <p style="color:var(--ink-soft); line-height:2;">
            • اختبار النموذجين على بيانات سريرية حقيقية للمقارنة.<br>
            • تجربة نموذج ثالث مثل XGBoost للتأكد من ثبات النتائج.<br>
            • دراسة أثر حذف عمود Total_Symptoms لمعرفة ما إذا كانت الأعمدة
              الأصلية وحدها كافية للوصول لنفس الدقة.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; color:var(--ink-soft); font-size:.75rem; margin-top:2rem; padding-bottom:1rem;">
        قسم أمراض القلب · مبادرة EarlyMed للكشف المبكر · لأغراض تعليمية فقط
    </div>
    """,
    unsafe_allow_html=True,
)
