import pandas as pd
import json , re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.offline as pyo
#import plotly.graph_objs as go
#pyo.init_notebook_mode(connected=True)
import plotly.express as px
import streamlit as st
    

def load_category_data(json_file_path):
    """Load the categorized resumes from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def prepare_training_data(category_data):
    """
    Prepare training data for TF-IDF vectorizer
    Returns: documents, category labels, and category to document mapping
    """
    documents = []
    categories = []
    category_docs = defaultdict(list)
    
    for category, resumes in category_data.items():
        for resume in resumes:
            documents.append(resume)
            categories.append(category)
            category_docs[category].append(resume)
    
    return documents, categories, category_docs

def find_top_categories(text_cv, json_file_path, top_n=5):
    """
    Find top N most similar categories to the given CV text
    
    Args:
        text_cv (str): The CV text to compare
        json_file_path (str): Path to the JSON file with categorized resumes
        top_n (int): Number of top categories to return
    
    Returns:
        list: Top N categories with their similarity scores
    """
    
    # Load category data
    category_data = load_category_data(json_file_path)
    
    # Prepare training data
    documents, categories, category_docs = prepare_training_data(category_data)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform the training documents
    X = vectorizer.fit_transform(documents)
    
    # Transform the input CV text
    cv_vector = vectorizer.transform([text_cv])
    
    # Calculate cosine similarity between CV and all documents
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    # Calculate average similarity per category
    category_scores = defaultdict(list)
    for i, similarity in enumerate(similarities):
        category_scores[categories[i]].append(similarity)
    
    # Get average similarity for each category
    category_avg_scores = {}
    for category, scores in category_scores.items():
        category_avg_scores[category] = np.mean(scores)
    
    # Sort categories by average similarity (descending)
    sorted_categories = sorted(
        category_avg_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    # Return top N categories
    result = []
    for category, score in sorted_categories:
        percentage = round(score, 2)  # Convert to percentage and round to 2 decimals
        result.append((category, percentage))
    
    return result,similarities,vectorizer

def find_top_categories_alternative(text_cv, json_file_path, top_n=5):
    """
    Alternative approach: Create representative text for each category
    and compare with input CV
    """
    
    category_data = load_category_data(json_file_path)
    
    # Create representative text for each category (concatenate all resumes)
    category_representatives = {}
    for category, resumes in category_data.items():
        # Combine all resumes in the category with some separation
        representative_text = " ".join(resumes)
        category_representatives[category] = representative_text
    
    # Prepare documents for vectorization
    documents = list(category_representatives.values())
    category_names = list(category_representatives.keys())
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform category representative texts
    X = vectorizer.fit_transform(documents)
    
    # Transform the input CV text
    cv_vector = vectorizer.transform([text_cv])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    # Pair categories with their similarity scores
    category_scores = list(zip(category_names, similarities))
    
    # Sort by similarity (descending)
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    return category_scores[:top_n]

def preprocess_text(text):
    """Enhanced text preprocessing"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def enhanced_find_top_categories(text_cv, json_file_path, top_n=5):
    """Enhanced version with better text preprocessing"""
    
    category_data = load_category_data(json_file_path)
    
    # Preprocess all texts
    preprocessed_cv = preprocess_text(text_cv)
    
    category_representatives = {}
    for category, resumes in category_data.items():
        # Preprocess and combine resumes
        preprocessed_resumes = [preprocess_text(resume) for resume in resumes]
        representative_text = " ".join(preprocessed_resumes)
        category_representatives[category] = representative_text
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=6000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85
    )
    
    documents = list(category_representatives.values())
    category_names = list(category_representatives.keys())
    
    X = vectorizer.fit_transform(documents)
    cv_vector = vectorizer.transform([preprocessed_cv])
    
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    category_scores = list(zip(category_names, similarities))
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    return category_scores[:top_n]

def print_formatted_results(results):
    """Print results in a nice formatted way"""
    percentage_results = []
    for i, (category, score) in enumerate(results, 1):
        percentage = round(float(score) * 100, 2)
        percentage_results.append((category, percentage))
        #print(f"{i}. {category}: {percentage}%")
        st.write(f"{i}. {category}: {percentage}%")

    
    return percentage_results



def visualize_cv_in_trained_space(similarities, documents, categories, input_cv_text, vectorizer=None):
    """
    Visualize input CV position within trained CV space using dimensionality reduction
    
    Args:
        similarities: Cosine similarity scores between input CV and all training documents
        documents: List of all training CV texts
        categories: List of categories for each training document
        input_cv_text: The input CV text to visualize
        vectorizer: Optional pre-fitted vectorizer (if None, creates new one)
    """
    
    # Import plotly offline for Jupyter notebook display

    # Create or use existing vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        training_vectors = vectorizer.fit_transform(documents)
    else:
        training_vectors = vectorizer.transform(documents)
    
    # Transform input CV using the same vectorizer
    input_cv_vector = vectorizer.transform([input_cv_text])
    
    # Combine all vectors for consistent dimensionality reduction
    all_vectors = np.vstack([training_vectors.toarray(), input_cv_vector.toarray()])
    
    # Apply dimensionality reduction
    #print("Reducing dimensions with t-SNE...")
    st.info("Reducing dimensions with t-SNE...")

    
    # First reduce with PCA for better t-SNE performance
    pca = PCA(n_components=min(50, all_vectors.shape[1]))
    reduced_pca = pca.fit_transform(all_vectors)
    
    # Use t-SNE for final 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(documents)-1))
    reduced_2d = tsne.fit_transform(reduced_pca)
    
    # Separate training points and input CV point
    training_points = reduced_2d[:-1]  # All except last point
    input_cv_point = reduced_2d[-1]    # Last point is input CV
    
    # Create DataFrame for training data
    plot_data = pd.DataFrame({
        'x': training_points[:, 0],
        'y': training_points[:, 1],
        'category': categories,
        'similarity': similarities,
        'hover_text': [f"Category: {cat}<br>Similarity: {sim:.3f}" 
                      for cat, sim in zip(categories, similarities)],
        'size': [sim * 20 + 5 for sim in similarities]  # Size based on similarity
    })
    
    # Create the main scatter plot for training data
    fig = px.scatter(
        plot_data, 
        x='x', 
        y='y',
        color='category',
        hover_data={'hover_text': True, 'x': False, 'y': False, 'category': False},
        size='size',
        title='Input CV Position in Trained CV Space',
        labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
        opacity=0.7
    )
    
    # Add the input CV as a special point
    fig.add_trace(go.Scatter(
        x=[input_cv_point[0]],
        y=[input_cv_point[1]],
        mode='markers+text',
        marker=dict(
            size=20,
            color='black',
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='Input CV',
        hoverinfo='text',
        hovertext='Your Input CV',
        text='Your CV',
        textposition='middle center'
    ))
    
    # Update layout
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        width=900,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    #print("Visualization ready! Showing plot...")
    
    # Use offline iplot for Jupyter notebook display
    #pyo.iplot(fig)
    st.success("Visualization ready!")
    st.plotly_chart(fig, use_container_width=True)
    
    return fig, reduced_2d

def find_missing_keywords(input_cv_text, target_category, category_data, vectorizer=None, top_n=10):
    """
    Find missing keywords in input CV that are important for a target category
    
    Args:
        input_cv_text: The input CV text
        target_category: The category you want to match better
        category_data: Dictionary of {category: [resumes]}
        vectorizer: Optional pre-fitted vectorizer
        top_n: Number of top missing keywords to return
    
    Returns:
        List of tuples (keyword, importance_score) that are missing
    """
    
    # Create or use existing vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        all_documents = []
        for resumes in category_data.values():
            all_documents.extend(resumes)
        vectorizer.fit(all_documents)
    
    # Get all resumes for the target category
    target_resumes = category_data.get(target_category, [])
    if not target_resumes:
        #print(f"No resumes found for category: {target_category}")
        st.warning(f"No resumes found for category: {target_category}")

        return []
    
    # Create a representative document for the target category
    target_combined = " ".join(target_resumes)
    
    # Transform both documents
    input_vector = vectorizer.transform([input_cv_text])
    target_vector = vectorizer.transform([target_combined])
    
    # Get feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate TF-IDF scores
    input_scores = input_vector.toarray()[0]
    target_scores = target_vector.toarray()[0]
    
    # Find keywords that are important in target but missing/weak in input
    missing_keywords = []
    for i, (input_score, target_score) in enumerate(zip(input_scores, target_scores)):
        if target_score > 0.01 and input_score < target_score * 0.3:  # Important in target but weak in input
            importance = target_score - input_score
            missing_keywords.append((feature_names[i], importance, target_score, input_score))
    
    # Sort by importance (difference)
    missing_keywords.sort(key=lambda x: x[1], reverse=True)
    
    return missing_keywords[:top_n]

def get_category_keyword_profile(target_category, category_data, vectorizer=None, top_n=15):
    """
    Get the most important keywords for a specific category
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        all_documents = []
        for resumes in category_data.values():
            all_documents.extend(resumes)
        vectorizer.fit(all_documents)
    
    target_resumes = category_data.get(target_category, [])
    if not target_resumes:
        return []
    
    target_combined = " ".join(target_resumes)
    target_vector = vectorizer.transform([target_combined])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords for this category
    scores = target_vector.toarray()[0]
    keyword_scores = [(feature_names[i], score) for i, score in enumerate(scores) if score > 0]
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]

def analyze_cv_gaps(input_cv_text, target_category, category_data, vectorizer=None):
    """
    Comprehensive analysis of what's missing in the CV for a target category
    """
    #print(f"=== CV Gap Analysis for '{target_category}' ===\n")
    st.subheader(f"CV Gap Analysis for '{target_category}'")

    
    # Get missing keywords
    missing_keywords = find_missing_keywords(input_cv_text, target_category, category_data, vectorizer)
    
    # Get category profile
    category_profile = get_category_keyword_profile(target_category, category_data, vectorizer)
    
    # Get current CV keywords
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        all_documents = []
        for resumes in category_data.values():
            all_documents.extend(resumes)
        vectorizer.fit(all_documents)
    
    input_vector = vectorizer.transform([input_cv_text])
    input_scores = input_vector.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    cv_keywords = [(feature_names[i], score) for i, score in enumerate(input_scores) if score > 0]
    cv_keywords.sort(key=lambda x: x[1], reverse=True)
    
    #print("📊 Top keywords in your CV:")
    st.markdown("### 📊 Top keywords in your CV:")

    for keyword, score in cv_keywords[:10]:
        #print(f"   {keyword}: {score:.4f}")
        st.write(f"- {keyword}: {score:.4f}")

    
    #print(f"\n🎯 Top keywords expected for '{target_category}':")
    st.markdown(f"### 🎯 Top keywords expected for '{target_category}':")
    for keyword, score in category_profile:
        print(f"   {keyword}: {score:.4f}")
    
    #print(f"\n🔍 Missing/weak keywords for '{target_category}':")
    st.markdown(f"### 🔍 Missing/weak keywords for '{target_category}':")
    if not missing_keywords:
        #print("   No significant missing keywords found! Your CV already matches well.")
        st.success(" No significant missing keywords found! Your CV already matches well.")
        return
    
    for i, (keyword, importance, target_score, input_score) in enumerate(missing_keywords, 1):
        #print(f"   {i}. {keyword}")
        #print(f"      Importance: {importance:.4f} (Target: {target_score:.4f}, Your CV: {input_score:.4f})")
        st.write(f"{i}. **{keyword}**")
        st.caption(f"Importance: {importance:.4f} (Target: {target_score:.4f}, Your CV: {input_score:.4f})")
    
    return missing_keywords

def suggest_improvements(input_cv_text, target_category, category_data, vectorizer=None):
    """
    Generate actionable suggestions to improve CV for target category
    """
    missing_keywords = find_missing_keywords(input_cv_text, target_category, category_data, vectorizer)
    
    if not missing_keywords:
        return ["Your CV already has strong alignment with this category!"]
    
    suggestions = []
    suggestions.append(f"To better match '{target_category}', consider adding:")
    
    # Group keywords by type for better suggestions
    technical_terms = []
    tools_tech = []
    methodologies = []
    soft_skills = []
    
    for keyword, importance, _, _ in missing_keywords:
        keyword_lower = keyword.lower()
        
        # Categorize keywords
        if any(tech in keyword_lower for tech in ['python', 'java', 'sql', 'c++', 'javascript', 'html', 'css']):
            technical_terms.append(keyword)
        elif any(tool in keyword_lower for tool in ['docker', 'aws', 'azure', 'git', 'jenkins', 'kubernetes']):
            tools_tech.append(keyword)
        elif any(method in keyword_lower for method in ['agile', 'scrum', 'devops', 'ci/cd', 'machine learning']):
            methodologies.append(keyword)
        else:
            soft_skills.append(keyword)
    
    if technical_terms:
        suggestions.append(f"• Technical skills: {', '.join(technical_terms[:5])}")
    if tools_tech:
        suggestions.append(f"• Tools & technologies: {', '.join(tools_tech[:5])}")
    if methodologies:
        suggestions.append(f"• Methodologies: {', '.join(methodologies[:5])}")
    if soft_skills:
        suggestions.append(f"• Soft skills: {', '.join(soft_skills[:3])}")
    
    suggestions.append("\n💡 Tips:")
    suggestions.append("- Add specific projects using these technologies")
    suggestions.append("- Include metrics and results achieved")
    suggestions.append("- Consider getting certifications in missing areas")
    suggestions.append("- Tailor your experience descriptions to highlight relevant skills")
    
    return suggestions
