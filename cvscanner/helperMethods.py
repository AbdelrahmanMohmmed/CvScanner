
import json
import requests
import os

API_KEY = "AIzaSyD90K_alA1ApxjWVLQFE_oNxbBMenBaRo8"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def generate_better_cv(job_title, cv_content, missing_skills):
    print("Generating a better CV version using an LLM...")

    prompt = f"""
    You are a professional career coach and CV writer. Your task is to rewrite
    the following CV section for a '{job_title}' position to make it more impactful and professional.

    Focus on the following points:
    1.  Improve the language to use action verbs and quantifiable results.
    2.  Incorporate the following missing skills into the new version: {', '.join(missing_skills)}.
    3.  Make the content concise and focused on the job title.

    Original CV section:
    ---
    {cv_content}
    ---

    Provide only the rewritten section. Do not include any introductory or concluding remarks.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {
            "parts": [{"text": "Act as a world-class career coach and professional CV writer."}]
        },
    }

    try:
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        
        # Check if the response contains content
        candidate = result.get("candidates", [])[0]
        if candidate and candidate.get("content") and candidate["content"].get("parts"):
            generated_text = candidate["content"]["parts"][0]["text"]
            
            # Extract and print sources for grounded responses (if any)
            sources = []
            grounding_metadata = candidate.get("groundingMetadata")
            if grounding_metadata and grounding_metadata.get("groundingAttributions"):
                sources = [
                    attr["web"]["uri"]
                    for attr in grounding_metadata["groundingAttributions"]
                    if "web" in attr
                ]
            if sources:
                print("\nSources used for grounding:")
                for source in sources:
                    print(f"- {source}")
            
            return generated_text
        else:
            return "Error: No text content found in the API response."
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API call: {e}")
        return "Failed to generate a better CV. Please check your API key and network connection."

def group_and_save_to_json(df, group_by, output_file):
    """
    Groups a DataFrame by a specified column, combines all other columns into text per row,
    collects them into a list, and saves as JSON.
    
    :param df: pandas DataFrame
    :param group_by: column name to group by (string)
    :param output_file: path to save the JSON file
    """
    grouped_data = {}

    for key, group in df.groupby(group_by):
        combined_list = []
        for _, row in group.iterrows():
            # Combine all columns (except group_by) into one string
            row_text = ", ".join(str(row[col]) for col in group.columns if col != group_by)
            combined_list.append(row_text)
        
        grouped_data[key] = combined_list

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=4, ensure_ascii=False)

    return grouped_data
