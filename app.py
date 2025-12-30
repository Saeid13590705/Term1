import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import math
from datetime import datetime

# تنظیمات صفحه
st.set_page_config(
    page_title="سیستم تحلیل نمرات مدرسه",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل فارسی پیشرفته
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        color: #2A5298;
        border-right: 5px solid #1E3C72;
        padding-right: 15px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 700;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 4px solid #1E3C72;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .success-card {
        border-top-color: #28a745;
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e9 100%);
    }
    
    .warning-card {
        border-top-color: #ffc107;
        background: linear-gradient(135deg, #fffbf0 0%, #fff3cd 100%);
    }
    
    .danger-card {
        border-top-color: #dc3545;
        background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
    }
    
    .info-card {
        border-top-color: #17a2b8;
        background: linear-gradient(135deg, #f0f9ff 0%, #e3f2fd 100%);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    
    .teacher-report {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }
    
    .highlight-box {
        background: #fff3cd;
        border-right: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# توابع محاسباتی
def calculate_iqr_statistics(data):
    """محاسبه آمار IQR برای یک سری داده"""
    if len(data) < 3:
        return None
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # محاسبه میانه
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # محاسبه چارک‌ها بر اساس روش دقیق
    if n % 2 == 1:  # تعداد فرد
        median_pos = n // 2
        lower_half = sorted_data[:median_pos]
        upper_half = sorted_data[median_pos + 1:]
    else:  # تعداد زوج
        mid_pos1 = n // 2 - 1
        mid_pos2 = n // 2
        lower_half = sorted_data[:mid_pos2]
        upper_half = sorted_data[mid_pos1 + 1:]
    
    # تابع میانه داخلی
    def calc_median(arr):
        if not arr:
            return None
        m = len(arr)
        if m % 2 == 1:
            return arr[m // 2]
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2
    
    q1 = calc_median(lower_half)
    q3 = calc_median(upper_half)
    
    if q1 is None or q3 is None:
        return None
    
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [x for x in sorted_data if x < lower_bound or x > upper_bound]
    
    return {
        'count': n,
        'mean': float(np.mean(data)),
        'median': float(median),
        'std': float(np.std(data)) if n > 1 else 0,
        'min': float(min(data)),
        'max': float(max(data)),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percent': (len(outliers) / n * 100) if n > 0 else 0
    }

def analyze_subject_scores(df, subject_name):
    """تحلیل نمرات یک درس خاص"""
    scores = df[subject_name].dropna().tolist()
    
    if not scores:
        return None
    
    stats = calculate_iqr_statistics(scores)
    if stats is None:
        return None
    
    # تحلیل کیفیت تدریس
    analysis = {
        'stats': stats,
        'grade_distribution': categorize_scores(scores),
        'weaknesses': identify_weaknesses(scores, subject_name),
        'strengths': identify_strengths(scores, subject_name),
        'recommendations': generate_recommendations(stats, subject_name)
    }
    
    return analysis

def categorize_scores(scores):
    """دسته‌بندی نمرات به ضعیف، متوسط، خوب، عالی"""
    categories = {
        'ضعیف (0-9)': len([s for s in scores if 0 <= s <= 9]),
        'قابل قبول (10-14)': len([s for s in scores if 10 <= s <= 14]),
        'خوب (15-17)': len([s for s in scores if 15 <= s <= 17]),
        'عالی (18-20)': len([s for s in scores if 18 <= s <= 20])
    }
    return categories

def identify_weaknesses(scores, subject_name):
    """شناسایی نقاط ضعف"""
    weaknesses = []
    
    if len([s for s in scores if s < 10]) / len(scores) > 0.3:
        weaknesses.append(f"تعداد زیاد دانش‌آموزان ضعیف (نمره زیر ۱۰)")
    
    if np.std(scores) > 6:
        weaknesses.append("پراکندگی زیاد نمرات (اختلاف سطح بالا)")
    
    if min(scores) == 0:
        weaknesses.append("وجود نمره صفر (نیاز به بررسی ویژه)")
    
    if len([s for s in scores if s < 5]) > 0:
        weaknesses.append("وجود نمرات بسیار پایین (زیر ۵)")
    
    return weaknesses

def identify_strengths(scores, subject_name):
    """شناسایی نقاط قوت"""
    strengths = []
    
    if np.mean(scores) > 15:
        strengths.append("میانگین کلاس عالی")
    
    if len([s for s in scores if s >= 18]) / len(scores) > 0.4:
        strengths.append("تعداد قابل توجه دانش‌آموزان ممتاز")
    
    if np.std(scores) < 4:
        strengths.append("همگنی مناسب کلاس")
    
    if min(scores) > 10:
        strengths.append("عدم وجود دانش‌آموز بسیار ضعیف")
    
    return strengths

def generate_recommendations(stats, subject_name):
    """تولید توصیه‌های آموزشی"""
    recommendations = []
    
    # بر اساس میانگین
    mean = stats['mean']
    if mean < 12:
        recommendations.append("🔴 **نیاز فوری**: برگزاری کلاس‌های جبرانی فشرده")
    elif mean < 15:
        recommendations.append("🟡 **نیاز متوسط**: افزایش تمرین‌های تکمیلی")
    else:
        recommendations.append("🟢 **وضعیت مطلوب**: ادامه رویه فعلی با افزودن چالش‌های بیشتر")
    
    # بر اساس پراکندگی
    if stats['std'] > 6:
        recommendations.append("🎯 **تدریس تفکیکی**: گروه‌بندی دانش‌آموزان بر اساس سطح")
    
    # بر اساس outliers
    if stats['outlier_count'] > 0:
        recommendations.append("👥 **حمایت ویژه**: توجه خاص به دانش‌آموزان outlier")
    
    # بر اساس IQR
    if stats['iqr'] > 8:
        recommendations.append("📊 **بازبینی روش**: بررسی تأثیر روش تدریس فعلی")
    
    return recommendations

def compare_classes(df, class1, class2, subject_name):
    """مقایسه دو کلاس در یک درس"""
    scores1 = df[df['کلاس'] == class1][subject_name].dropna().tolist()
    scores2 = df[df['کلاس'] == class2][subject_name].dropna().tolist()
    
    if not scores1 or not scores2:
        return None
    
    stats1 = calculate_iqr_statistics(scores1)
    stats2 = calculate_iqr_statistics(scores2)
    
    comparison = {
        'class1': {
            'name': class1,
            'stats': stats1,
            'analysis': analyze_subject_scores(df[df['کلاس'] == class1], subject_name)
        },
        'class2': {
            'name': class2,
            'stats': stats2,
            'analysis': analyze_subject_scores(df[df['کلاس'] == class2], subject_name)
        },
        'comparison_points': compare_statistics(stats1, stats2)
    }
    
    return comparison

def compare_statistics(stats1, stats2):
    """مقایسه آماری دو مجموعه داده"""
    points = []
    
    # مقایسه میانگین
    diff_mean = stats2['mean'] - stats1['mean']
    if diff_mean > 2:
        points.append(f"کلاس دوم به طور قابل توجهی میانگین بالاتری دارد (+{diff_mean:.1f})")
    elif diff_mean < -2:
        points.append(f"کلاس اول میانگین بالاتری دارد ({abs(diff_mean):.1f} واحد)")
    else:
        points.append("تفاوت معنی‌داری در میانگین وجود ندارد")
    
    # مقایسه پراکندگی
    if stats2['std'] < stats1['std'] - 1:
        points.append(f"کلاس دوم همگن‌تر است (انحراف معیار کمتر)")
    elif stats2['std'] > stats1['std'] + 1:
        points.append(f"کلاس اول همگن‌تر است")
    
    # مقایسه میانه
    diff_median = stats2['median'] - stats1['median']
    if abs(diff_median) > 2:
        points.append(f"تفاوت قابل توجه در میانه: {diff_median:.1f} واحد")
    
    # مقایسه تعداد ضعیف‌ها
    weak1 = stats1['mean'] < 10
    weak2 = stats2['mean'] < 10
    if weak1 and not weak2:
        points.append("کلاس اول نیاز فوری به مداخله دارد")
    elif not weak1 and weak2:
        points.append("کلاس دوم نیاز فوری به مداخله دارد")
    
    return points

def generate_teacher_report(df, subject_column, teacher_name=""):
    """تولید گزارش جامع برای معلم"""
    scores = df[subject_column].dropna().tolist()
    
    if not scores:
        return None
    
    stats = calculate_iqr_statistics(scores)
    analysis = analyze_subject_scores(df, subject_column)
    
    report = {
        'teacher': teacher_name,
        'subject': subject_column,
        'date': datetime.now().strftime("%Y/%m/%d"),
        'summary': generate_summary(stats, analysis),
        'detailed_analysis': analysis,
        'action_items': generate_action_items(stats, analysis),
        'success_stories': identify_success_stories(df, subject_column),
        'concerns': identify_concerns(df, subject_column)
    }
    
    return report

def generate_summary(stats, analysis):
    """خلاصه گزارش"""
    summary = []
    
    mean = stats['mean']
    if mean >= 16:
        summary.append("🎉 **عملکرد عالی**: میانگین کلاس در سطح ممتاز")
    elif mean >= 14:
        summary.append("✅ **عملکرد خوب**: میانگین کلاس قابل قبول")
    elif mean >= 12:
        summary.append("⚠️ **نیاز به بهبود**: میانگین کلاس نیاز به ارتقا دارد")
    else:
        summary.append("🚨 **نیاز به مداخله فوری**: میانگین کلاس بسیار پایین")
    
    if stats['outlier_percent'] > 20:
        summary.append(f"⚠️ **تعداد زیاد outlier**: {stats['outlier_percent']:.1f}% دانش‌آموزان خارج از محدوده عادی")
    
    if stats['std'] > 6:
        summary.append("📊 **پراکندگی بالا**: اختلاف سطح دانش‌آموزان زیاد است")
    
    return summary

def generate_action_items(stats, analysis):
    """اقدامات لازم"""
    actions = []
    
    # اقدامات بر اساس میانگین
    if stats['mean'] < 12:
        actions.append({
            'priority': 'بالا',
            'action': 'برگزاری کلاس جبرانی فشرده',
            'deadline': 'فوری',
            'responsible': 'معلم'
        })
    
    # اقدامات برای outliers
    if stats['outlier_count'] > 0:
        actions.append({
            'priority': 'متوسط',
            'action': 'جلسات مشاوره فردی با دانش‌آموزان outlier',
            'deadline': '۲ هفته',
            'responsible': 'معلم + مشاور'
        })
    
    # اقدامات برای پراکندگی
    if stats['std'] > 5:
        actions.append({
            'priority': 'متوسط',
            'action': 'تدریس تفکیکی و گروه‌بندی',
            'deadline': '۱ ماه',
            'responsible': 'معلم'
        })
    
    return actions

def identify_success_stories(df, subject_column):
    """شناسایی موفقیت‌ها"""
    success = []
    df_sorted = df.sort_values(subject_column, ascending=False)
    
    # برترین دانش‌آموزان
    top_students = df_sorted.head(3)[['نام', 'نام خانوادگی', subject_column]].to_dict('records')
    if top_students:
        success.append(f"**برترین دانش‌آموزان**: {', '.join([f'{s['نام']} {s['نام خانوادگی']} ({s[subject_column]})' for s in top_students])}")
    
    # بیشترین پیشرفت (اگر داده تاریخی داریم)
    if 'معدل' in df.columns:
        high_gpa = df[df[subject_column] >= 18]
        if len(high_gpa) > 0:
            success.append(f"**هماهنگی با معدل**: {len(high_gpa)} دانش‌آموز هم در این درس و هم در معدل عالی هستند")
    
    return success

def identify_concerns(df, subject_column):
    """شناسایی نگرانی‌ها"""
    concerns = []
    
    # دانش‌آموزان با نمره زیر ۱۰
    weak_students = df[df[subject_column] < 10][['نام', 'نام خانوادگی', subject_column, 'کلاس']]
    if len(weak_students) > 3:
        concerns.append(f"**تعداد زیاد ضعیف**: {len(weak_students)} دانش‌آموز نمره زیر ۱۰ دارند")
    
    # نمرات صفر
    zero_scores = df[df[subject_column] == 0]
    if len(zero_scores) > 0:
        concerns.append(f"**نمره صفر**: {len(zero_scores)} دانش‌آموز نمره صفر گرفته‌اند")
    
    # عدم مشارکت (اگر ستون حضور داریم)
    if 'انضباط' in df.columns:
        low_discipline = df[(df[subject_column] < 10) & (df['انضباط'] < 15)]
        if len(low_discipline) > 0:
            concerns.append(f"**مشکل انضباطی و درسی**: {len(low_discipline)} دانش‌آموز هم نمره پایین و هم انضباط ضعیف دارند")
    
    return concerns

# رابط کاربری اصلی
def main():
    # هدر اصلی
    st.markdown("""
    <div class="main-title">
        <h1 style="margin:0; font-size: 2.5rem;">🏫 سیستم هوشمند تحلیل نمرات مدرسه</h1>
        <p style="margin:10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        ابزار مدیریت آموزشی برای مدیران و معلمان | تحلیل آماری پیشرفته نمرات
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # سایدبار
    with st.sidebar:
        st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
        st.header("⚙️ تنظیمات تحلیل")
        
        # آپلود فایل
        uploaded_file = st.file_uploader("📁 فایل اکسل نمرات را آپلود کنید", 
                                        type=['xlsx', 'xls'])
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ فایل با موفقیت خوانده شد")
                st.info(f"تعداد رکوردها: {len(df)}")
                
                # نمایش ستون‌ها
                if st.checkbox("نمایش ستون‌های فایل"):
                    st.write(df.columns.tolist())
                    
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")
                df = None
        else:
            st.info("👈 لطفاً فایل نمونه یا فایل خود را آپلود کنید")
            df = None
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # اگر فایل آپلود شده
    if 'df' in locals() and df is not None:
        # تب‌های مختلف
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 تحلیل کلی", 
            "👨‍🏫 گزارش معلم", 
            "📈 مقایسه کلاس‌ها", 
            "🎯 شناسایی مشکلات", 
            "💾 خروجی گزارش"
        ])
        
        with tab1:
            st.markdown('<h3 class="sub-title">تحلیل کلی تمام دروس</h3>', unsafe_allow_html=True)
            
            # انتخاب دروس برای تحلیل
            subject_columns = st.multiselect(
                "دروس مورد نظر برای تحلیل را انتخاب کنید:",
                options=[col for col in df.columns if col not in ['ردیف', 'کلاس', 'نام', 'نام خانوادگی', 'معدل', 'متنمعدل', 'حروفی', 'انضباط', 'جمع']],
                default=['ریاضی', 'علوم', 'ادبیات فارسی']
            )
            
            if subject_columns:
                cols = st.columns(len(subject_columns))
                for idx, subject in enumerate(subject_columns):
                    with cols[idx]:
                        analysis = analyze_subject_scores(df, subject)
                        if analysis:
                            stats = analysis['stats']
                            
                            # کارت متریک
                            card_class = "success-card" if stats['mean'] >= 15 else "warning-card" if stats['mean'] >= 12 else "danger-card"
                            st.markdown(f'<div class="metric-card {card_class} rtl-text">', unsafe_allow_html=True)
                            st.metric(subject, f"{stats['mean']:.1f}", 
                                    f"±{stats['std']:.1f} STD")
                            st.caption(f"تعداد: {stats['count']} | میانه: {stats['median']:.1f}")
                            st.caption(f"ضعیف: {analysis['grade_distribution']['ضعیف (0-9)']} نفر")
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # نمودار مقایسه‌ای
                st.markdown('<h4 class="sub-title">مقایسه دروس</h4>', unsafe_allow_html=True)
                
                fig_data = []
                for subject in subject_columns:
                    scores = df[subject].dropna().tolist()
                    if scores:
                        fig_data.append({
                            'درس': subject,
                            'میانگین': np.mean(scores),
                            'میانه': np.median(scores),
                            'انحراف معیار': np.std(scores),
                            'حداقل': min(scores),
                            'حداکثر': max(scores)
                        })
                
                if fig_data:
                    df_compare = pd.DataFrame(fig_data)
                    fig = px.bar(df_compare, x='درس', y='میانگین',
                                title='میانگین نمرات دروس مختلف',
                                color='میانگین',
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<h3 class="sub-title">گزارش تخصصی برای معلم</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                selected_subject = st.selectbox(
                    "درس مورد نظر:",
                    options=[col for col in df.columns if col not in ['ردیف', 'کلاس', 'نام', 'نام خانوادگی', 'معدل', 'متنمعدل', 'حروفی', 'انضباط', 'جمع']]
                )
            
            with col2:
                teacher_name = st.text_input("نام معلم:", value="")
            
            if selected_subject:
                report = generate_teacher_report(df, selected_subject, teacher_name)
                
                if report:
                    # نمایش گزارش در کارت‌های زیبا
                    st.markdown('<div class="teacher-report rtl-text">', unsafe_allow_html=True)
                    st.subheader(f"📋 گزارش درس {selected_subject}")
                    if teacher_name:
                        st.write(f"**معلم:** {teacher_name}")
                    st.write(f"**تاریخ گزارش:** {report['date']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # خلاصه
                    st.markdown('<div class="highlight-box rtl-text">', unsafe_allow_html=True)
                    st.write("### 📊 خلاصه عملکرد")
                    for item in report['summary']:
                        st.write(f"- {item}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # آمار دقیق
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("میانگین", f"{report['detailed_analysis']['stats']['mean']:.2f}")
                        st.metric("حداقل", f"{report['detailed_analysis']['stats']['min']:.2f}")
                    with col2:
                        st.metric("میانه", f"{report['detailed_analysis']['stats']['median']:.2f}")
                        st.metric("حداکثر", f"{report['detailed_analysis']['stats']['max']:.2f}")
                    with col3:
                        st.metric("انحراف معیار", f"{report['detailed_analysis']['stats']['std']:.2f}")
                        st.metric("IQR", f"{report['detailed_analysis']['stats']['iqr']:.2f}")
                    
                    # توزیع نمرات
                    st.markdown('<h4 class="sub-title">توزیع نمرات</h4>', unsafe_allow_html=True)
                    dist_df = pd.DataFrame.from_dict(
                        report['detailed_analysis']['grade_distribution'], 
                        orient='index', 
                        columns=['تعداد']
                    )
                    dist_df['درصد'] = (dist_df['تعداد'] / report['detailed_analysis']['stats']['count'] * 100).round(1)
                    st.dataframe(dist_df, use_container_width=True)
                    
                    # نمودار هیستوگرام
                    scores = df[selected_subject].dropna().tolist()
                    fig = px.histogram(x=scores, nbins=20, 
                                      title=f'توزیع نمرات درس {selected_subject}',
                                      labels={'x': 'نمره', 'y': 'تعداد دانش‌آموز'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # اقدامات لازم
                    st.markdown('<h4 class="sub-title">📝 اقدامات پیشنهادی</h4>', unsafe_allow_html=True)
                    actions_df = pd.DataFrame(report['action_items'])
                    st.dataframe(actions_df, use_container_width=True)
                    
                    # موفقیت‌ها و نگرانی‌ها
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="success-card rtl-text">', unsafe_allow_html=True)
                        st.write("### 🎉 نقاط قوت")
                        for item in report['success_stories']:
                            st.write(f"- {item}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="warning-card rtl-text">', unsafe_allow_html=True)
                        st.write("### ⚠️ نقاط ضعف")
                        for item in report['concerns']:
                            st.write(f"- {item}")
                        for weakness in report['detailed_analysis']['weaknesses']:
                            st.write(f"- {weakness}")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h3 class="sub-title">مقایسه عملکرد کلاس‌ها</h3>', unsafe_allow_html=True)
            
            # اگر ستون کلاس وجود دارد
            if 'کلاس' in df.columns:
                classes = df['کلاس'].unique().tolist()
                
                if len(classes) >= 2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        class1 = st.selectbox("کلاس اول:", classes)
                    with col2:
                        class2 = st.selectbox("کلاس دوم:", [c for c in classes if c != class1])
                    with col3:
                        compare_subject = st.selectbox(
                            "درس مورد مقایسه:",
                            options=[col for col in df.columns if col not in ['ردیف', 'کلاس', 'نام', 'نام خانوادگی', 'معدل', 'متنمعدل', 'حروفی', 'انضباط', 'جمع']]
                        )
                    
                    if class1 and class2 and compare_subject:
                        comparison = compare_classes(df, class1, class2, compare_subject)
                        
                        if comparison:
                            # نمایش نتایج مقایسه
                            st.markdown('<div class="info-card rtl-text">', unsafe_allow_html=True)
                            st.write(f"### 📊 مقایسه {class1} و {class2} در {compare_subject}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{class1}:**")
                                st.metric("میانگین", f"{comparison['class1']['stats']['mean']:.2f}")
                                st.metric("میانه", f"{comparison['class1']['stats']['median']:.2f}")
                                st.metric("انحراف معیار", f"{comparison['class1']['stats']['std']:.2f}")
                            
                            with col2:
                                st.write(f"**{class2}:**")
                                st.metric("میانگین", f"{comparison['class2']['stats']['mean']:.2f}")
                                st.metric("میانه", f"{comparison['class2']['stats']['median']:.2f}")
                                st.metric("انحراف معیار", f"{comparison['class2']['stats']['std']:.2f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # نکات مقایسه
                            st.markdown('<div class="highlight-box rtl-text">', unsafe_allow_html=True)
                            st.write("### 🔍 نتایج مقایسه")
                            for point in comparison['comparison_points']:
                                st.write(f"- {point}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # نمودار مقایسه‌ای
                            fig = go.Figure()
                            
                            # Boxplot برای کلاس اول
                            scores1 = df[df['کلاس'] == class1][compare_subject].dropna().tolist()
                            fig.add_trace(go.Box(
                                y=scores1,
                                name=class1,
                                boxpoints='outliers',
                                marker_color='blue'
                            ))
                            
                            # Boxplot برای کلاس دوم
                            scores2 = df[df['کلاس'] == class2][compare_subject].dropna().tolist()
                            fig.add_trace(go.Box(
                                y=scores2,
                                name=class2,
                                boxpoints='outliers',
                                marker_color='red'
                            ))
                            
                            fig.update_layout(
                                title=f'مقایسه Boxplot {compare_subject}',
                                yaxis_title='نمره',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("حداقل دو کلاس برای مقایسه نیاز است")
            else:
                st.warning("ستون 'کلاس' در فایل یافت نشد")
        
        with tab4:
            st.markdown('<h3 class="sub-title">شناسایی سیستماتیک مشکلات</h3>', unsafe_allow_html=True)
            
            # شناسایی دروس مشکل‌دار
            problem_subjects = []
            subject_columns = [col for col in df.columns if col not in ['ردیف', 'کلاس', 'نام', 'نام خانوادگی', 'معدل', 'متنمعدل', 'حروفی', 'انضباط', 'جمع']]
            
            for subject in subject_columns:
                analysis = analyze_subject_scores(df, subject)
                if analysis:
                    stats = analysis['stats']
                    weaknesses = analysis['weaknesses']
                    
                    if stats['mean'] < 12 or len(weaknesses) > 2:
                        problem_subjects.append({
                            'درس': subject,
                            'میانگین': stats['mean'],
                            'مشکلات': weaknesses,
                            'تعداد ضعیف': analysis['grade_distribution']['ضعیف (0-9)'],
                            'اولویت': 'بالا' if stats['mean'] < 10 else 'متوسط'
                        })
            
            if problem_subjects:
                st.markdown('<div class="danger-card rtl-text">', unsafe_allow_html=True)
                st.write("### 🚨 دروس نیازمند توجه فوری")
                problems_df = pd.DataFrame(problem_subjects)
                st.dataframe(problems_df.sort_values('میانگین'), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # شناسایی دانش‌آموزان مشکل‌دار
                st.markdown('<h4 class="sub-title">👥 دانش‌آموزان نیازمند حمایت ویژه</h4>', unsafe_allow_html=True)
                
                weak_students = []
                for _, row in df.iterrows():
                    low_scores = []
                    for subject in subject_columns:
                        if pd.notna(row[subject]) and row[subject] < 10:
                            low_scores.append(f"{subject}: {row[subject]}")
                    
                    if len(low_scores) >= 3:  # اگر در ۳ درس یا بیشتر ضعیف باشد
                        weak_students.append({
                            'نام': f"{row['نام']} {row['نام خانوادگی']}",
                            'کلاس': row['کلاس'] if 'کلاس' in row else '-',
                            'تعداد دروس ضعیف': len(low_scores),
                            'دروس ضعیف': ', '.join(low_scores[:3]) + ('...' if len(low_scores) > 3 else '')
                        })
                
                if weak_students:
                    weak_df = pd.DataFrame(weak_students)
                    st.dataframe(weak_df.sort_values('تعداد دروس ضعیف', ascending=False), 
                                use_container_width=True)
                else:
                    st.info("✅ دانش‌آموز با مشکل جدی شناسایی نشد")
            else:
                st.success("🎉 هیچ درس مشکل‌داری شناسایی نشد!")
        
        with tab5:
            st.markdown('<h3 class="sub-title">خروجی گزارش‌ها</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "نوع گزارش:",
                    ["گزارش کلی مدرسه", "گزارش درسی خاص", "گزارش مقایسه کلاس‌ها", "گزارش مشکلات"]
                )
            
            with col2:
                if report_type == "گزارش درسی خاص":
                    report_subject = st.selectbox(
                        "درس:",
                        options=[col for col in df.columns if col not in ['ردیف', 'کلاس', 'نام', 'نام خانوادگی', 'معدل', 'متنمعدل', 'حروفی', 'انضباط', 'جمع']]
                    )
                elif report_type == "گزارش مقایسه کلاس‌ها":
                    if 'کلاس' in df.columns:
                        classes = df['کلاس'].unique().tolist()
                        report_class1 = st.selectbox("کلاس اول:", classes)
                        report_class2 = st.selectbox("کلاس دوم:", [c for c in classes if c != report_class1])
            
            if st.button("📄 تولید گزارش PDF"):
                # اینجا می‌توانید از کتابخانه‌هایی مثل reportlab یا weasyprint استفاده کنید
                # برای سادگی، یک خروجی HTML ایجاد می‌کنیم
                
                import base64
                
                # ایجاد گزارش HTML ساده
                html_report = """
                <!DOCTYPE html>
                <html dir="rtl">
                <head>
                    <meta charset="UTF-8">
                    <title>گزارش تحلیلی نمرات</title>
                    <style>
                        body { font-family: 'Vazirmatn', sans-serif; padding: 20px; }
                        .header { text-align: center; background: #1E3C72; color: white; padding: 20px; border-radius: 10px; }
                        .metric { background: #f8f9fa; padding: 15px; margin: 10px 0; border-right: 5px solid #007bff; }
                        .warning { background: #fff3cd; border-color: #ffc107; }
                        .danger { background: #f8d7da; border-color: #dc3545; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>گزارش تحلیلی نمرات مدرسه</h1>
                        <p>تاریخ تولید: """ + datetime.now().strftime("%Y/%m/%d") + """</p>
                    </div>
                    <h2>خلاصه آماری</h2>
                """
                
                # اضافه کردن آمار
                for subject in subject_columns[:5]:  # فقط ۵ درس اول
                    analysis = analyze_subject_scores(df, subject)
                    if analysis:
                        stats = analysis['stats']
                        html_report += f"""
                        <div class="metric">
                            <h3>{subject}</h3>
                            <p>میانگین: {stats['mean']:.2f} | میانه: {stats['median']:.2f}</p>
                            <p>تعداد دانش‌آموز: {stats['count']} | انحراف معیار: {stats['std']:.2f}</p>
                        </div>
                        """
                
                html_report += "</body></html>"
                
                # ایجاد فایل HTML قابل دانلود
                b64 = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="school_report.html">📥 دانلود گزارش HTML</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # همچنین امکان ذخیره در اکسل
                if st.button("📊 ذخیره آمار در اکسل"):
                    # ایجاد DataFrame از آمار
                    stats_list = []
                    for subject in subject_columns:
                        analysis = analyze_subject_scores(df, subject)
                        if analysis:
                            stats = analysis['stats']
                            stats_list.append({
                                'درس': subject,
                                'میانگین': stats['mean'],
                                'میانه': stats['median'],
                                'انحراف معیار': stats['std'],
                                'حداقل': stats['min'],
                                'حداکثر': stats['max'],
                                'تعداد': stats['count']
                            })
                    
                    stats_df = pd.DataFrame(stats_list)
                    
                    # ایجاد خروجی Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, sheet_name='آمار دروس', index=False)
                    
                    output.seek(0)
                    
                    # دکمه دانلود
                    st.download_button(
                        label="📥 دانلود فایل اکسل",
                        data=output,
                        file_name="school_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    else:
        # صفحه راهنمای اولیه
        st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 راهنمای استفاده از سیستم
        
        ### ۱. آماده‌سازی فایل اکسل
        - فایل اکسل باید شامل ستون‌های زیر باشد:
          - کلاس (مثال: هشتم/۱)
          - نام و نام خانوادگی دانش‌آموزان
          - ستون‌های نمرات دروس مختلف
          - ستون معدل (اختیاری)
        
        ### ۲. ویژگی‌های سیستم
        
        #### 📊 تحلیل کلی
        - نمایش آمار توصیفی همه دروس
        - نمودارهای مقایسه‌ای
        - شناسایی دروس قوی و ضعیف
        
        #### 👨‍🏫 گزارش معلم
        - گزارش تخصصی برای هر معلم
        - تحلیل عمیق نقاط قوت و ضعف
        - پیشنهادات بهبود تدریس
        
        #### 📈 مقایسه کلاس‌ها
        - مقایسه عملکرد دو کلاس در یک درس
        - شناسایی بهترین روش‌های تدریس
        - اشتراک‌گذاری تجربیات موفق
        
        #### 🎯 شناسایی مشکلات
        - شناسایی سیستماتیک دانش‌آموزان نیازمند حمایت
        - کشف دروس مشکل‌دار
        - اولویت‌بندی مداخلات آموزشی
        
        ### ۳. خروجی‌های سیستم
        - گزارش HTML قابل چاپ
        - فایل اکسل با آمار کامل
        - نمودارهای تعاملی
        
        ### ۴. نمونه فایل
        """, unsafe_allow_html=True)
        
        # ایجاد یک نمونه DataFrame
        sample_data = {
            'کلاس': ['هشتم/۱', 'هشتم/۱', 'هشتم/۲', 'هشتم/۲'],
            'نام': ['علی', 'رضا', 'سارا', 'نازنین'],
            'نام خانوادگی': ['محمدی', 'احمدی', 'کریمی', 'حسینی'],
            'ریاضی': [18, 12, 20, 15],
            'علوم': [17, 14, 19, 16],
            'ادبیات فارسی': [19, 16, 18, 17]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
