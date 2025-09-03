import json
import requests
import os,sys
import pandas as pd
from helperMethods import *

if __name__ == "__main__":
    # job_title = "Data Analyst"
    # existing_cv_content = "5+ years of experience in data analysis, working with large datasets to extract insights and build reports. Skilled in using Python for data cleaning and manipulation."
    # missing_skills_list = ["Statistical Analysis", "Machine Learning", "ETL"]

    # better_cv_text = generate_better_cv(job_title, existing_cv_content, missing_skills_list)

    # print("\n--- Generated 'Better' CV Version ---")
    # print(better_cv_text)
    df = pd.read_csv("/home/jax/CvScanner/data/processed/cleanedV4.csv",index_col=0)
    group_and_save_to_json(df, "name", "/home/jax/CvScanner/models/grouped_Data2.json")
