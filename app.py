# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
import PyPDF2


from cvscanner.modeling import DeepLearningModel
from cvscanner.modeling import NLP_methods as nlp
from cvscanner.modeling import machine_learning as ml

# -------------------
JSON_FILE = "models/grouping.json"       # for NLP
DL_MODEL_PATH = "models/best_cv_classifier.pth"  # deep learning model
CLEANED_DATA = "data/processed/cleanedV2.csv"  # for ML/EDA
# -------------------

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Download the model from Google Drive 
if not os.path.exists(DL_MODEL_PATH):
    file_id = "1lf3ggGMJHN-z75Nlkk_15QnvoxgexOJ4" 
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, DL_MODEL_PATH, quiet=False)

# Streamlit page setup
st.set_page_config(page_title="Smart Career Guidance", layout="wide")
st.title("Smart Career Guidance & Recruitment System")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Upload & Analyze CV", "NLP Insights", "ML Predictions", "EDA Dashboard"]
)

# ------------------- Tab 1 -------------------
with tab1:
    st.subheader("Upload your CV or paste text")
    uploaded_file = st.file_uploader("Upload CV (txt/pdf)", type=["txt", "pdf"])
    user_input = st.text_area("Or paste your CV text here:")

    # Function to extract text from PDF or TXT
    def extract_text(file):
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        else:
            return file.read().decode("utf-8")

    if st.button("Analyze CV"):
        text = ""
        if uploaded_file is not None:
            text = extract_text(uploaded_file)
        elif user_input.strip():
            text = user_input

        if not text.strip():
            st.warning("Please provide some CV text.")
        else:
            st.success("CV text loaded successfully!")

            # --- NLP Analysis ---
            st.subheader("NLP Analysis")
            results, similarities, vectorizer = nlp.find_top_categories(
                text, JSON_FILE, top_n=5
            )
            st.write("**Top categories (NLP):**")
            for category, score in results:
                st.write(f"- {category} ({score:.2%})")

            best_category = results[0][0]
            category_data = nlp.load_category_data(JSON_FILE)
            missing = nlp.find_missing_keywords(
                text, best_category, category_data, vectorizer
            )
            st.write(f"**Missing skills for {best_category}:**")
            if missing:
                for word, importance, _, _ in missing:
                    st.write(f"- {word} (importance: {importance:.2f})")
            else:
                st.success("No major skill gaps found!")

            # --- Deep Learning Prediction ---
            st.subheader("Deep Learning Model (BERT)")
            model, label_encoder, tokenizer = DeepLearningModel.load_trained_model(DL_MODEL_PATH)
            pred_cat, confidence, top_preds = DeepLearningModel.predict_cv_category(
                text, model, label_encoder, tokenizer
            )
            st.write(f"**Predicted Career Path:** {pred_cat} ({confidence*100:.2f}%)")
            st.write("**Top predictions:**")
            for cat, conf in top_preds:
                st.write(f"- {cat}: {conf*100:.2f}%")

# ------------------- Tab 2 -------------------
with tab2:
    st.subheader("Explore NLP Models")
    example_text = st.text_area("Enter example CV text for NLP analysis:")
    if st.button("Run NLP Example"):
        if example_text.strip():
            results = nlp.enhanced_find_top_categories(example_text, JSON_FILE, top_n=5)
            nlp.print_formatted_results(results)
            #st.write(results)

# ------------------- Tab 3 -------------------
with tab3:
    st.subheader("ML-based Career Path Prediction")
    student_skills = st.text_input("Enter skills (comma separated):", "python, sql, machine learning")
    if st.button("Predict with ML"):
        clf = ml.clf  # trained SVM model in ML_model.py
        pred = clf.predict([student_skills])[0]
        st.success(f"Predicted Career Path: {pred}")


# ------------------- Tab 4 -------------------
with tab4:
    st.subheader("EDA & Insights from Data")
    df = pd.read_csv(CLEANED_DATA)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Job Categories Distribution")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(y=df["Category"], order=df["Category"].value_counts().index, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Top Skills")
        all_skills = [s for lst in df["Skills"].dropna() for s in eval(lst)]
        skill_counts = pd.Series(all_skills).value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=skill_counts.values, y=skill_counts.index, ax=ax)
        st.pyplot(fig)

    # ------------------- WordCloud -------------------
    st.write("Skills WordCloud")
    if all_skills: 
        from wordcloud import WordCloud

        skill_text = " ".join(all_skills)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(skill_text)

        fig_wc, ax_wc = plt.subplots(figsize=(15,7))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("No skills found to generate WordCloud.")
