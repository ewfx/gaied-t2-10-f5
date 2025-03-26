
from email_classification import process_email
import glob
import json

def process_email_batch(input_folder, output_file):
    results = []
    files = glob.glob(f"{input_folder}/*.*") 
    
    for file in files:
        if file.lower().endswith(('.pdf', '.eml', '.txt', '.doc', '.docx')):
            result = process_email(file)
            if result:
                results.extend(result)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)