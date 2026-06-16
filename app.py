
import streamlit as st
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import re

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}

SPAM_INDICATORS = {'free', 'winner', 'cash', 'prize', 'urgent', 'act now', 'limited', 'offer', 'click', 'buy', 'order', 'call now', 'text', 'win', 'congratulations', 'selected', 'claim', 'credit', 'loan', 'debt', 'weight', 'lose', 'act', 'important', 'alert', 'warning', 'verify', 'account', 'suspended', 'bank', 'nigerian', 'million', 'dollars', 'viagra', 'medicine', 'pills', 'discount', 'cheap', 'save', 'time', 'expires', 'today', 'only', 'special', 'promotion', 'deal', 'bonus', 'gift', 'reward', 'click here', 'link', 'website', 'visit', 'subscribe', 'unsubscribe', 'marketing', 'advertisement', 'promo', 'sale', 'price', 'cost', 'pay', 'payment', 'invoice', 'bill', 'charge', 'refund', 'money', 'dollar', 'pound', 'euro', 'usd', 'investment', 'profit', 'income', 'earn', 'make money', 'work from home', 'job', 'opportunity', 'career', 'hiring', 'recruitment', 'resume', 'cv', 'apply', 'register', 'sign up', 'join', 'membership', 'subscription', 'trial', 'free trial', 'demo'}

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s']', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def tokenize(text):
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return ' '.join(tokens)

def get_spam_word_count(text):
    cleaned = clean_text(text)
    tokens = set(cleaned.split())
    return len(tokens & SPAM_INDICATORS)

def validate_input(text, max_length=10000):
    if not text or not isinstance(text, str):
        return False, "Please enter valid text. Input cannot be empty."
    if len(text.strip()) == 0:
        return False, "Input cannot be empty or whitespace only."
    if len(text) > max_length:
        return False, f"Input exceeds maximum length of {max_length} characters."
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if len(tokens) < 2:
        return False, "Input must contain at least 2 meaningful words."
    return True, None

def get_top_features(vectorizer, model, text, n_features=10):
    import numpy as np
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        coef = model.coef_[0]
        feature_scores = np.array(X.toarray()[0]) * coef
    elif hasattr(model, 'feature_log_prob_'):
        log_prob_spam = model.feature_log_prob_[1]
        log_prob_ham = model.feature_log_prob_[0]
        feature_scores = np.array(X.toarray()[0]) * (log_prob_spam - log_prob_ham)
    else:
        return {"features": [], "scores": [], "directions": []}
    non_zero_indices = X.nonzero()[1]
    if len(non_zero_indices) == 0:
        return {"features": [], "scores": [], "directions": []}
    text_feature_scores = [(feature_names[i], feature_scores[i]) for i in non_zero_indices]
    text_feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    top_n = text_feature_scores[:n_features]
    return {"features": [item[0] for item in top_n], "scores": [round(float(item[1]), 4) for item in top_n], "directions": ["spam" if item[1] > 0 else "ham" for item in top_n]}

def generate_rule_explanation(top_features, threshold=0.01):
    rules = []
    for feat, score, direction in zip(top_features["features"], top_features["scores"], top_features["directions"]):
        if abs(score) < threshold:
            continue
        confidence = min(abs(score) * 100, 95)
        action = "increases" if direction == "spam" else "decreases"
        rules.append(f"IF '{feat}' appears -> {action} spam likelihood by {confidence:.1f}%")
    return rules[:5]

def generate_natural_language_explanation(text, label, probability, top_features, model_name):
    cleaned = clean_text(text)
    tokens = set(cleaned.split())
    spam_words_found = tokens & SPAM_INDICATORS
    confidence_pct = probability * 100
    parts = []
    if label == "spam":
        parts.append(f"This email was classified as **SPAM** with {confidence_pct:.1f}% confidence.")
    else:
        parts.append(f"This email was classified as **LEGITIMATE (HAM)** with {confidence_pct:.1f}% confidence.")
    parts.append(f"The {model_name} model made this decision based on the following analysis:")
    if top_features["features"]:
        spam_features = [f for f, d in zip(top_features["features"], top_features["directions"]) if d == "spam"]
        ham_features = [f for f, d in zip(top_features["features"], top_features["directions"]) if d == "ham"]
        if spam_features and label == "spam":
            parts.append(f"\n**Spam indicators:** The words '{', '.join(spam_features[:3])}' strongly suggest spam.")
        if ham_features and label == "ham":
            parts.append(f"\n**Legitimate indicators:** The words '{', '.join(ham_features[:3])}' are common in real emails.")
    if spam_words_found:
        parts.append(f"\n**Promotional language:** Found {len(spam_words_found)} spam-related words: '{', '.join(list(spam_words_found)[:5])}'.")
    word_count = len(cleaned.split())
    if word_count < 10:
        parts.append(f"\n**Length:** Very short ({word_count} words) -- spammers often send brief, urgent messages.")
    elif word_count > 100:
        parts.append(f"\n**Length:** Longer message ({word_count} words) -- typical of legitimate communication.")
    if label == "spam":
        parts.append(f"\n**Recommendation:** This email shows multiple spam characteristics. Do not click links or share personal info.")
    else:
        parts.append(f"\n**Recommendation:** This email appears legitimate. Always verify sender identity before sharing sensitive data.")
    return "\n".join(parts)

def generate_feature_importance_summary(top_features):
    lines = ["**Top Contributing Features:**\n"]
    for feat, score, direction in zip(top_features["features"], top_features["scores"], top_features["directions"]):
        emoji = "🔴" if direction == "spam" else "🟢"
        impact = "strong" if abs(score) > 0.5 else "moderate" if abs(score) > 0.1 else "weak"
        lines.append(f"{emoji} **{feat}** -- {impact} {direction} indicator (score: {score:.3f})")
    return "\n".join(lines)

import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_confidence_donut(probability, label):
    spam_prob = probability * 100
    ham_prob = (1 - probability) * 100
    colors = ['#FF6B6B', '#4ECDC4'] if label == 'spam' else ['#4ECDC4', '#FF6B6B']
    fig = go.Figure(data=[go.Pie(values=[spam_prob, ham_prob], labels=['Spam', 'Ham'], hole=0.6, marker_colors=colors, textinfo='label+percent', textfont_size=14)])
    fig.update_layout(title_text=f"Classification Confidence: {label.upper()}", title_x=0.5, title_font_size=18, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), height=400, margin=dict(t=60, b=40, l=40, r=40))
    center_color = '#FF6B6B' if label == 'spam' else '#4ECDC4'
    fig.add_annotation(text=f"{spam_prob:.1f}%<br>Spam", x=0.5, y=0.5, font_size=20, font_color=center_color, showarrow=False)
    return fig

def create_feature_bar_chart(top_features):
    features = top_features["features"]
    scores = top_features["scores"]
    directions = top_features["directions"]
    colors = ['#FF6B6B' if d == 'spam' else '#4ECDC4' for d in directions]
    fig = go.Figure(data=[go.Bar(x=scores, y=features, orientation='h', marker_color=colors, text=[f"{s:+.3f}" for s in scores], textposition='outside')])
    fig.update_layout(title_text="Top Features Influencing Decision", title_x=0.5, title_font_size=16, xaxis_title="Impact Score (positive = spam, negative = ham)", yaxis_title="", height=400, margin=dict(t=60, b=40, l=150, r=80), plot_bgcolor='rgba(240,240,240,0.5)')
    fig.add_vline(x=0, line_width=2, line_color="gray", line_dash="dash")
    return fig

def create_wordcloud_image(text, spam_words):
    def color_func(word, *args, **kwargs):
        return '#FF6B6B' if word.lower() in [w.lower() for w in spam_words] else '#4ECDC4'
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, relative_scaling=0.5, color_func=color_func).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud (Red = Spam Indicators)', fontsize=14, pad=20)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def create_metrics_comparison(metrics):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR']
    model_names = list(metrics.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    fig = go.Figure()
    for i, model in enumerate(model_names):
        values = [metrics[model].get('accuracy', 0), metrics[model].get('precision', 0), metrics[model].get('recall', 0), metrics[model].get('f1_score', 0), metrics[model].get('false_positive_rate', 0)]
        fig.add_trace(go.Bar(name=model, x=metric_names, y=values, marker_color=colors[i % len(colors)], text=[f"{v:.3f}" for v in values], textposition='outside'))
    fig.update_layout(title_text="Model Performance Comparison", title_x=0.5, title_font_size=18, xaxis_title="Metric", yaxis_title="Score", barmode='group', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(t=100, b=40, l=60, r=40), plot_bgcolor='rgba(240,240,240,0.5)', yaxis=dict(range=[0, 1.1]))
    return fig

def create_confusion_matrix_heatmap(cm, labels, title):
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted ' + l for l in labels], y=['Actual ' + l for l in labels], text=cm, texttemplate="%{text}", textfont={"size": 16}, colorscale='RdYlBu'))
    fig.update_layout(title_text=title, title_x=0.5, title_font_size=16, height=400, width=500, margin=dict(t=60, b=40, l=100, r=40))
    return fig

def create_batch_results_table(results_df):
    fig = go.Figure(data=[go.Table(header=dict(values=list(results_df.columns), fill_color='#45B7D1', align='left', font=dict(color='white', size=12)), cells=dict(values=[results_df[col] for col in results_df.columns], fill_color=[['#F0F0F0' if i % 2 == 0 else 'white' for i in range(len(results_df))]], align='left', font=dict(size=11), height=30))])
    fig.update_layout(title_text="Batch Classification Results", title_x=0.5, height=min(600, 100 + len(results_df) * 35), margin=dict(t=60, b=20, l=20, r=20))
    return fig

def create_roc_curve(fpr_list, tpr_list, auc_scores):
    fig = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (model_name, fpr, tpr) in enumerate(zip(auc_scores.keys(), fpr_list, tpr_list)):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC = {auc_scores[model_name]:.3f})", line=dict(color=colors[i % len(colors)], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(color='gray', width=1, dash='dash')))
    fig.update_layout(title_text="ROC Curve Comparison", title_x=0.5, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin=dict(t=60, b=80, l=60, r=40))
    return fig

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown('<style>.main-header { font-size: 40px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; } .sub-header { font-size: 18px; color: #666; text-align: center; margin-bottom: 32px; } .spam-badge { background-color: #FF6B6B; color: white; padding: 8px 16px; border-radius: 8px; font-size: 24px; font-weight: bold; } .ham-badge { background-color: #4ECDC4; color: white; padding: 8px 16px; border-radius: 8px; font-size: 24px; font-weight: bold; }</style>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    vectorizer = pickle.load(open("/content/tfidf_vectorizer.pkl", "rb"))
    nb_model = pickle.load(open("/content/nb_model.pkl", "rb"))
    lr_model = pickle.load(open("/content/lr_model.pkl", "rb"))
    eval_results = pickle.load(open("/content/evaluation_results.pkl", "rb"))
    return vectorizer, nb_model, lr_model, eval_results

def predict_email(model, vectorizer, text):
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    return ("spam" if prediction == 1 else "ham"), probabilities[1], processed

vectorizer, nb_model, lr_model, eval_results = load_models()

st.markdown('<div class="main-header">🛡️ SpamGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable Email Spam Detection with Machine Learning</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select Model", ["Naive Bayes", "Logistic Regression"])
    model = nb_model if model_choice == "Naive Bayes" else lr_model
    st.divider()
    st.subheader("📧 Quick Test Samples")
    sample_emails = {
        "Select a sample...": "",
        "🚨 Spam: Prize Winner": "Congratulations! You've won a $1000 gift card. Call now to claim your prize! This is a limited time offer.",
        "🚨 Spam: Urgent Account": "URGENT: Your account will be suspended. Click here to verify immediately. Failure to act will result in permanent closure.",
        "✅ Ham: Lunch Meeting": "Hey, are we still meeting for lunch tomorrow at 12? Let me know if you need to reschedule.",
        "✅ Ham: Project Update": "Thanks for the update on the project. Let me review the documents and get back to you by end of day.",
        "🚨 Spam: Free Entry": "Free entry to win a car! Text WIN to 55555 now! You've been selected from millions of entries.",
        "✅ Ham: Grocery List": "Can you pick up some milk on your way home? We also need bread and eggs for dinner.",
    }
    selected_sample = st.selectbox("Load sample email", list(sample_emails.keys()))
    sample_text = sample_emails[selected_sample]
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("**SpamGuard AI** classifies emails as spam or legitimate using ML. Features: real-time classification, interactive visualizations, explainable predictions, model comparison.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Email Analysis", "📁 Batch Analysis", "📊 Model Comparison"])

with tab1:
    st.header("Single Email Analysis")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📥 Input")
        email_text = st.text_area("Enter email text:", value=sample_text, height=200, placeholder="Paste your email content here...")
        uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
        if uploaded_file is not None:
            email_text = uploaded_file.read().decode('utf-8')
            st.text_area("File content:", value=email_text, height=150, disabled=True)
        analyze_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with col2:
        st.subheader("📤 Results")
        if analyze_btn:
            is_valid, error_msg = validate_input(email_text)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                with st.spinner("Analyzing..."):
                    time.sleep(0.5)
                    start_time = time.time()
                    label, spam_prob, processed_text = predict_email(model, vectorizer, email_text)
                    inference_time = (time.time() - start_time) * 1000
                    top_features = get_top_features(vectorizer, model, processed_text, n_features=10)
                    spam_count = get_spam_word_count(email_text)
                    if label == "spam":
                        st.markdown('<div class="spam-badge">🚨 SPAM</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ham-badge">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{spam_prob*100:.1f}%", delta=f"Inference: {inference_time:.1f}ms")
                    st.info(f"📊 Found **{spam_count}** spam-indicator words")
        else:
            st.info("👈 Enter an email and click **Analyze Email**")

    if analyze_btn and is_valid:
        st.divider()
        viz_col1, viz_col2 = st.columns([1, 1])
        with viz_col1:
            st.subheader("📊 Confidence Visualization")
            st.plotly_chart(create_confidence_donut(spam_prob, label), use_container_width=True)
        with viz_col2:
            st.subheader("🔑 Top Features")
            if top_features["features"]:
                st.plotly_chart(create_feature_bar_chart(top_features), use_container_width=True)
            else:
                st.warning("No significant features detected")
        st.subheader("☁️ Word Cloud")
        spam_words_found = list(set(preprocess_text(email_text).split()) & SPAM_INDICATORS)
        st.image(create_wordcloud_image(processed_text, spam_words_found), use_container_width=True)
        st.divider()
        st.header("🧠 Explainability")
        exp_col1, exp_col2 = st.columns([1, 1])
        with exp_col1:
            st.subheader("📋 Feature Attribution")
            st.markdown(generate_feature_importance_summary(top_features))
            st.subheader("📜 Rule Extraction")
            rules = generate_rule_explanation(top_features)
            if rules:
                for rule in rules:
                    st.markdown(f"• {rule}")
            else:
                st.info("No strong rules extracted")
        with exp_col2:
            st.subheader("💬 Natural Language Explanation")
            nl_explanation = generate_natural_language_explanation(email_text, label, spam_prob if label == "spam" else 1 - spam_prob, top_features, model_choice)
            st.markdown(nl_explanation)
        st.divider()
        st.subheader("🔧 Intermediate Processing Steps")
        with st.expander("Click to see how the text was processed"):
            st.markdown("**Step 1: Raw Input**")
            st.text(email_text[:500] + ("..." if len(email_text) > 500 else ""))
            st.markdown("**Step 2: Cleaned Text**")
            cleaned = preprocess_text(email_text)
            st.text(cleaned[:500] + ("..." if len(cleaned) > 500 else ""))
            st.markdown("**Step 3: Tokenization**")
            tokens = cleaned.split()
            st.write(f"Total tokens: {len(tokens)} | Unique: {len(set(tokens))}")
            st.write("Sample:", tokens[:20])
            st.markdown("**Step 4: TF-IDF Vectorization**")
            X = vectorizer.transform([cleaned])
            st.write(f"Non-zero features: {X.nnz} | Vector shape: {X.shape}")

with tab2:
    st.header("Batch Email Analysis")
    st.markdown("Upload a CSV with `text` column (required) and `label` column (optional).")
    batch_file = st.file_uploader("Upload batch CSV:", type=['csv'], key="batch")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            if 'text' not in batch_df.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                st.success(f"✅ Loaded {len(batch_df)} emails")
                with st.expander("Preview"):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                if st.button("🚀 Run Batch Analysis", type="primary", key="run_batch"):
                    with st.spinner(f"Analyzing {len(batch_df)} emails..."):
                        results = []
                        for idx, row in batch_df.iterrows():
                            text = str(row['text'])
                            label, spam_prob, processed = predict_email(model, vectorizer, text)
                            result = {
                                'ID': idx + 1,
                                'Text_Preview': text[:100] + "..." if len(text) > 100 else text,
                                'Prediction': label.upper(),
                                'Spam_Probability': round(spam_prob * 100, 2),
                                'Spam_Words': get_spam_word_count(text)
                            }
                            if 'label' in batch_df.columns:
                                true_label = str(row['label']).lower()
                                result['True_Label'] = true_label.upper()
                                result['Correct'] = (label == true_label)
                            results.append(result)
                        results_df = pd.DataFrame(results)
                        st.subheader("📊 Batch Results")
                        st.plotly_chart(create_batch_results_table(results_df), use_container_width=True)
                        st.subheader("📈 Summary")
                        spam_count = sum(results_df['Prediction'] == 'SPAM')
                        ham_count = sum(results_df['Prediction'] == 'HAM')
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total", len(results_df))
                        c2.metric("Spam", spam_count, f"{spam_count/len(results_df)*100:.1f}%")
                        c3.metric("Ham", ham_count, f"{ham_count/len(results_df)*100:.1f}%")
                        if 'Correct' in results_df.columns:
                            st.metric("Accuracy", f"{results_df['Correct'].mean()*100:.1f}%")
                            y_true = [1 if l == 'SPAM' else 0 for l in results_df['True_Label']]
                            y_pred = [1 if l == 'SPAM' else 0 for l in results_df['Prediction']]
                            cm = confusion_matrix(y_true, y_pred)
                            st.subheader("Confusion Matrix")
                            st.plotly_chart(create_confusion_matrix_heatmap(cm, ['Ham', 'Spam'], "Batch Confusion Matrix"), use_container_width=True)
                        csv = results_df.to_csv(index=False)
                        st.download_button("⬇️ Download Results", csv, "spamguard_batch_results.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload a CSV to analyze multiple emails")

with tab3:
    st.header("Model Performance Comparison")
    if eval_results:
        metrics = eval_results['metrics']
        roc_data = eval_results['roc']
        st.subheader("📊 Metrics")
        st.plotly_chart(create_metrics_comparison(metrics), use_container_width=True)
        st.subheader("📋 Detailed Table")
        metrics_data = []
        for model_name, model_metrics in metrics.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{model_metrics['accuracy']:.4f}",
                'Precision': f"{model_metrics['precision']:.4f}",
                'Recall': f"{model_metrics['recall']:.4f}",
                'F1-Score': f"{model_metrics['f1_score']:.4f}",
                'FPR': f"{model_metrics['false_positive_rate']:.4f}"
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        st.subheader("🎯 Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("**Naive Bayes**")
            st.plotly_chart(create_confusion_matrix_heatmap(metrics['Naive Bayes']['confusion_matrix'], ['Ham', 'Spam'], "Naive Bayes"), use_container_width=True)
        with cm_col2:
            st.markdown("**Logistic Regression**")
            st.plotly_chart(create_confusion_matrix_heatmap(metrics['Logistic Regression']['confusion_matrix'], ['Ham', 'Spam'], "Logistic Regression"), use_container_width=True)
        st.subheader("📈 ROC Curves")
        fpr_list = [roc_data['fpr']['Naive Bayes'], roc_data['fpr']['Logistic Regression']]
        tpr_list = [roc_data['tpr']['Naive Bayes'], roc_data['tpr']['Logistic Regression']]
        st.plotly_chart(create_roc_curve(fpr_list, tpr_list, roc_data['auc']), use_container_width=True)
        st.subheader("🧠 Model Descriptions")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Naive Bayes**: Probabilistic classifier based on Bayes' theorem. Very fast, works well with high-dimensional text data. Strong independence assumption.")
        with d2:
            st.markdown("**Logistic Regression**: Linear model with logistic function. Highly interpretable via feature coefficients. Good baseline with regularization.")
    else:
        st.warning("Evaluation results not available.")

st.divider()
st.markdown('<div style="text-align: center; color: #666; font-size: 14px;">SpamGuard AI -- Built for AI Lab Project</div>', unsafe_allow_html=True)
