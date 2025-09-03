import pandas as pd
import json


df = pd.read_csv('/home/jax/CvScanner/data/processed/cleanedV2.csv')
json_file = '/home/jax/CvScanner/models/grouping.json'

category_resumes = {}

for category, group in df.groupby('Category'):
    category_resumes[category] = group['Cleaned_Resume'].tolist()

with open(json_file, 'w') as f:
    json.dump(category_resumes, f, ensure_ascii=False, indent=2)