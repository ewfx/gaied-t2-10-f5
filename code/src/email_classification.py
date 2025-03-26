import os
import re
import json
import time
import logging
import numpy as np
import requests
from email import policy
from email.parser import BytesParser
import pdfplumber
import docx
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# 1. CONFIGURATION SETUP
# ========================
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="email_classification.log",
)

# Initialize DeepSeek enabled flag
DEEPSEEK_ENABLED = True if os.getenv("DEEPSEEK_API_KEY") else False
print(f"DeepSeek enabled: {DEEPSEEK_ENABLED}")  # Debug output

# ========================
# 2. MODEL INITIALIZATION
# ========================
try:
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    exit(1)

# Constants
VALID_CATEGORIES = {
    'Adjustment', 'AU Transfer', 'Closing Notice', 'Commitment Change',
    'Fee Payment', 'Money Movement Inbound', 'Money Movement Outbound'
}

SUBTYPE_MAPPING = {
    'Money Movement Inbound': ['Principal', 'Interest', 'Principal + Interest', 'Principal + Interest + Fee'],
    'Money Movement Outbound': ['Timebound', 'Foreign Currency'],
    'AU Transfer': ['Reallocation Fees', 'Amendment Fees', 'Reallocation Principal'],
    'Closing Notice': ['Cashless Roll', 'Decrease', 'Increase'],
    'Fee Payment': ['Ongoing Fee', 'Letter of Credit Fee']
}

# ============================
# 3. CORE FUNCTION DEFINITIONS
# ============================

def extract_text_with_ocr(file_path):
    """Extract text from image-based documents using OCR"""
    try:
        if file_path.endswith(('.pdf', '.doc', '.docx')):
            try:
                import pytesseract
                from PIL import Image
                if file_path.endswith('.pdf'):
                    with pdfplumber.open(file_path) as pdf:
                        images = []
                        for page in pdf.pages:
                            if page.images:
                                for img in page.images:
                                    images.append(Image.frombytes('RGB', (img['width'], img['height']), img['stream'].get_data()))
                        if images:
                            return "\n".join(pytesseract.image_to_string(img) for img in images)
            except ImportError:
                logging.warning("OCR dependencies not installed. Falling back to regular extraction.")
            except Exception as e:
                logging.warning(f"OCR attempt failed: {str(e)}")
        return ""
    except Exception as e:
        logging.error(f"OCR extraction failed: {str(e)}")
        return ""

def extract_text_from_file(file_path):
    """Extract text from supported file types with enhanced error handling and OCR fallback"""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""

        # First try regular extraction
        text = ""
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
            if not text:  # Fallback to OCR if no text found
                text = extract_text_with_ocr(file_path)
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text).strip()
        elif file_path.endswith(".eml"):
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
                # Extract the main email body text
                text = msg.get_body(preferencelist=('plain',)).get_content().strip()
                
                # Extract and process attachments
                attachments = extract_eml_attachments(msg)
                for attachment_path in attachments:
                    attachment_text = extract_text_from_file(attachment_path)
                    if attachment_text:
                        text += f"\n\n[Attachment: {os.path.basename(attachment_path)}]\n{attachment_text}"
        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return ""
        
        return text if text else extract_text_with_ocr(file_path)  # Final OCR fallback
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_eml_metadata(file_path):
    """Extract metadata from EML files"""
    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
            return {
                'from': msg['from'],
                'to': msg['to'],
                'subject': msg['subject'],
                'date': msg['date']
            }
        # Extract attachments
        attachments = extract_eml_attachments(msg)
        metadata['attachments'] = attachments
        return metadata
    except Exception as e:
        logging.error(f"Error extracting EML metadata: {str(e)}")
        return {}

def extract_eml_attachments(msg):
    """Extract attachments from an EML file"""
    attachments = []
    try:
        for part in msg.iter_attachments():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition:
                file_name = part.get_filename()
                if file_name:
                    file_data = part.get_payload(decode=True)
                    file_path = os.path.join("attachments", file_name)
                    
                    # Ensure the attachments directory exists
                    os.makedirs("attachments", exist_ok=True)
                    
                    # Save the attachment to the attachments directory
                    with open(file_path, "wb") as f:
                        f.write(file_data)
                    
                    # Add the saved file path to the attachments list
                    attachments.append(file_path)
    except Exception as e:
        logging.error(f"Error extracting attachments: {str(e)}")
    return attachments    

def validate_subtype(subtype, category):
    """Validate subtype against allowed values for category"""
    if category not in SUBTYPE_MAPPING:
        return None
    return subtype if subtype in SUBTYPE_MAPPING[category] else None

def assign_to_team(classification, entities):
    """Assign the request to appropriate team based on classification and entities"""
    rules = {
        'Money Movement Inbound': 'Payment Processing Team',
        'Money Movement Outbound': 'Wire Transfer Team',
        'Adjustment': 'Loan Servicing Team',
        'AU Transfer': 'Account Management Team',
        'Closing Notice': 'Loan Operations Team',
        'Fee Payment': 'Billing Team',
        'Commitment Change': 'Underwriting Team'
    }
    
    # Special case for large amounts
    if 'MONEY' in entities:
        amounts = []
        for amount in entities['MONEY']:
            try:
                amounts.append(float(amount.replace('$','').replace(',','')))
            except ValueError:
                continue
        if amounts and max(amounts) > 50000:
            return 'Senior Operations Team'
    
    return rules.get(classification['label'], 'General Servicing Team')

def query_deepseek(prompt, max_retries=2):
    """Robust DeepSeek API query with proper error handling"""
    global DEEPSEEK_ENABLED
    
    if not DEEPSEEK_ENABLED:
        return None

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DeepSeek API key not configured")
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are an AI text classifier."},
            {"role": "user", "content": prompt}
            ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 402:
                logging.error("Payment required - disabling DeepSeek")
                DEEPSEEK_ENABLED = False
                return None
            response.raise_for_status()
            return getClassificationJson(response)
            #return response.json()
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(1 * (attempt + 1))
    
    return None

def getClassificationJson(response : requests.Response):
    response_data = json.loads(response.text)
    if "choices" not in response_data or len(response_data["choices"]) == 0:
                logging.error("No choices found in the response.")
                return None

    assistant_message = response_data["choices"][0]["message"]["content"]

    # Extract the JSON string from the code block
    json_match = re.search(r"```json\n({.*?})\n```", assistant_message, re.DOTALL)
    if not json_match:
        logging.error("No JSON content found in the assistant's message.")
        return None

    json_content = json_match.group(1)
    classification_data = json.loads(json_content)

    # Validate the response structure
    required_keys = ["category", "subtype", "confidence"]
    if not all(key in classification_data for key in required_keys):
                logging.error(f"Missing required keys in response: {classification_data}")
                return None

    confidence = classification_data["confidence"]
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logging.error(f"Invalid confidence level: {confidence}")
                return None

    return classification_data


def classify_with_deepseek(text):
    prompt = """You are a financial email classifier. Return JSON with these EXACT fields:
    {
        "category": "Money Movement Inbound|Adjustment|etc",
        "subtype": "Principal|Interest|etc or null",
        "confidence": 0.95,
        "reasoning": "Detailed explanation of classification",
        "entities": {
            "amount": "$15,000",
            "date": "25FEB21",
            "account": "Account123",
            "loan_id": "123456",
            "current_rate": "5.5%",
            "requested_rate": "4.5%"
        }
    }

    RULES:
    1. category MUST be one of: """ + ', '.join(VALID_CATEGORIES) + """
    2. For interest rate change requests, use "Adjustment" category
    3. subtype MUST be one of the allowed values for the category. Allowed subtypes is STRICTLY based on category
       and can be found in the dictionary : {json.dumps(SUBTYPE_MAPPING, indent=2)} 
    4. For money transfers, include amount/date/account
    5. For rate changes, include loan_id/current_rate/requested_rate
    6. reasoning MUST explain your decision

    Email Content:
    """ + text

    response = query_deepseek(prompt)
    #if not response or 'choices' not in response:
     #   return None

    try:
       # result = json.loads(response['choices'][0]['message']['content'])
        result = response
        # Strict validation
        """ if result['category'] not in VALID_CATEGORIES:
            raise ValueError("Invalid category")
            
        if not isinstance(result.get('entities', {}), dict):
            raise ValueError("Entities must be a dictionary")
            
        if 'reasoning' not in result:
            raise ValueError("Missing reasoning field") """
            
        return {
            'label': result['category'],
            'subtype': validate_subtype(result.get('subtype'), result['category']),
            'score': min(max(float(result.get('confidence', 0.5)), 0.0), 1.0),
            'reasoning': result.get('reasoning', 'No reasoning provided'),
            'llm_entities': result.get('entities', {})
        }

    except Exception as e:
        logging.error(f"LLM response parsing failed: {str(e)}")
        return None

def classify_with_fine_tuned_model(text):
    """Reliable fallback classification"""
    try:
        classifier = pipeline(
            "text-classification",
            model="./fine-tuned-model",
            tokenizer="./fine-tuned-model"
        )
        result = classifier(text)[0]
        
        # Improved fallback logic
        label = 'Money Movement Inbound' if result['label'] == 'LABEL_0' else 'Adjustment'
        
        # Infer subtype from text
        subtype = None
        if label == 'Money Movement Inbound':
            if 'principal' in text.lower() and 'interest' in text.lower():
                subtype = 'Principal+Interest'
            elif 'principal' in text.lower():
                subtype = 'Principal'
            elif 'interest' in text.lower():
                subtype = 'Interest'
        
        return {
            'label': label,
            'score': result['score'],
            'subtype': subtype,
            'reasoning': 'Classified by local model',
            'llm_entities': {}
        }
    except Exception as e:
        logging.error(f"Local model failed: {str(e)}")
        return {
            'label': 'Other',
            'score': 0.0,
            'subtype': None,
            'reasoning': 'Classification failed',
            'llm_entities': {}
        }

def classify_email(text):
    """Main classification function with forced LLM priority"""
    if not text.strip():
        return {
            'label': 'Other',
            'score': 0.0,
            'subtype': None,
            'reasoning': 'Empty text',
            'llm_entities': {}
        }
    
    # Step 1: Use the local fine-tuned model
    local_result = classify_with_fine_tuned_model(text)
    if local_result['score'] >= 0.95:
        # If confidence is >= 95%, return the local model's output
        logging.debug("Using local model. Local score is : " + str(local_result['score']))
        return local_result
    
    # Step 2: Fallback to LLM if local confidence is low
    if DEEPSEEK_ENABLED:
        logging.debug("Using LLM as Local score is below treshold : " + str(local_result['score']))
        llm_result = classify_with_deepseek(text)
    
        if llm_result:
            # Step 3: Train the local model using the LLM's output
            train_local_model(text, llm_result)
            return llm_result
    
    # Step 4: If both fail, return a default response
    return {
        'label': 'Other',
        'score': 0.0,
        'subtype': None,
        'reasoning': 'Classification failed',
        'llm_entities': {}
    }
   
def train_local_model(text, llm_result):
    """
    Train the local fine-tuned model using the LLM's output.
    """
    try:
        # Prepare the training data
        training_data = {
            'text': text,
            'label': llm_result['label'],
            'subtype': llm_result.get('subtype', None)
        }
        
        # Save the training data to a file (or database) for later use
        with open('training_data.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_data) + '\n')
        
        logging.info(f"Training data added: {training_data}")
        
        # Optionally, trigger a training pipeline here
        # Example: train_fine_tuned_model(training_data)
    except Exception as e:
        logging.error(f"Failed to train local model: {str(e)}")

def detect_duplicates(text, previous_emails, threshold=0.85):
        """Enhanced duplicate detection with better preview"""
        if not previous_emails:
          return None,0, None

   
        all_embeddings = sentence_model.encode(previous_emails + [text], convert_to_tensor=True)
        similarities = cosine_similarity([all_embeddings[-1]], all_embeddings[:-1])
        
        max_sim_idx = np.argmax(similarities)
        max_sim_score = similarities[0, max_sim_idx]

        if max_sim_score >= threshold:
            original_text = previous_emails[max_sim_idx]
            change_analysis = analyze_changes_with_llm(original_text, text)
            return original_text, max_sim_score, change_analysis
        return None, 0, None

def analyze_changes_with_llm(original_text, new_text):
    """Analyze changes between duplicate emails using LLM"""
    prompt = f"""
    You are an AI assistant. Compare the following two emails and identify:
    1. If there are any important changes (e.g., amount, interest rate, mood, or criticality).
    2. If the new email is just a follow-up or thank-you note.
    3. Provide a summary of the changes or intent.

    Original Email:
    {original_text}

    New Email:
    {new_text}

    Return a JSON object with the following fields:
    {{
        "important_changes": true/false,
        "changes_summary": "Summary of changes or intent",
        "intent": "Follow-up/Thank-you/Other"
    }}
    """
    response = query_deepseek(prompt)
    if response:
        try:
            return response  # Assuming the response is already a parsed JSON object
        except Exception as e:
            logging.error(f"Failed to parse LLM response for change analysis: {str(e)}")
    return {
        "important_changes": False,
        "changes_summary": "No significant changes detected",
        "intent": "Other"
    }

def extract_entities(text):
    """Comprehensive entity extraction with financial patterns"""
    try:
        doc = nlp(text)
        entities = {ent.label_: [] for ent in doc.ents}
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Enhanced financial patterns
        amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
        if amounts:
            entities["MONEY"] = amounts
            
        dates = re.findall(r'\b\d{1,2}[A-Z]{3}\d{2,4}\b', text)
        if dates:
            entities["DATE"] = dates
            
        account_numbers = re.findall(r'Account(?: #|:)?\s*([A-Z0-9-]+)', text, re.I)
        if account_numbers:
            entities["ACCOUNT"] = account_numbers
            
        loan_ids = re.findall(r'Loan (?:ID|Number)\s*:\s*([A-Z0-9-]+)', text, re.I)
        if loan_ids:
            entities["LOAN_ID"] = loan_ids
            
        rates = re.findall(r'(?:Rate|Interest)\s*:\s*([\d.]+%?)', text, re.I)
        if rates:
            entities["INTEREST_RATE"] = rates
            
        return {k: v for k, v in entities.items() if v}
    except Exception as e:
        logging.error(f"Entity extraction failed: {str(e)}")
        return {}

def determine_priority(text, classification):
    """Enhanced priority detection with amount thresholds"""
    text_lower = text.lower()
    
    # Immediate priority keywords
    if any(word in text_lower for word in ['urgent', 'asap', 'immediately']):
        return 'critical'
    
    amounts = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
    if amounts:
        try:
            max_amount = max(float(amt.replace('$','').replace(',','')) for amt in amounts)
            if max_amount >= 10000: return 'high'
            if max_amount >= 5000: return 'medium'
        except ValueError:
            pass
    
    # Category-based defaults
    return {
        'Money Movement Inbound': 'high',
        'Money Movement Outbound': 'high',
        'Adjustment': 'medium',
        'Fee Payment': 'medium'
    }.get(classification['label'], 'low')

def split_multi_requests(text):
    """Improved multi-request detection"""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    split_phrases = ['also', 'additionally', 'furthermore', 'second request', 'another request']
    split_points = [i for i, s in enumerate(sentences) 
                   if any(phrase in s.lower() for phrase in split_phrases)]
    
    if not split_points:
        return [text]
    
    parts = []
    start = 0
    for point in split_points:
        part = ' '.join(sentences[start:point])
        if part.strip():
            parts.append(part)
        start = point
    
    remaining = ' '.join(sentences[start:])
    if remaining.strip():
        parts.append(remaining)
    
    return parts

def process_email(file_path, previous_emails=None):
    """Complete email processing pipeline"""
    previous_emails = previous_emails or []
    text = extract_text_from_file(file_path)
    if not text:
        return None
    
    metadata = {}
    if file_path.endswith(".eml"):
        metadata = extract_eml_metadata(file_path)
    
    results = []
    for part in split_multi_requests(text):
        classification = classify_email(part)
        duplicate_of, similarity, change_analysis = detect_duplicates(part, previous_emails)
        spacy_entities = extract_entities(part)
        priority = determine_priority(part, classification)
        assigned_team = assign_to_team(classification, spacy_entities)
        
        # Combine entities from both spaCy and LLM
        all_entities = {**spacy_entities}
        if classification.get('llm_entities'):
            all_entities['LLM_Entities'] = classification['llm_entities']
        
        results.append({
            'file_path': file_path,
            'metadata': metadata,
            'text': part,
            'classification': classification,
            'duplicate': bool(duplicate_of),
            'duplicate_of': duplicate_of if duplicate_of else None,
            'similarity_score': similarity if duplicate_of else None,
            'change_analysis': change_analysis if duplicate_of else None,
            'entities': all_entities,
            'priority': priority,
            'assigned_team': assigned_team,
            'processing_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return results

def load_previous_emails(file_path="previous_emails.json"):
    """Load previous emails from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load previous emails: {str(e)}")
    return []

def save_previous_emails(previous_emails, file_path="previous_emails.json"):
    """Save previous emails to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(previous_emails, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save previous emails: {str(e)}")

def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy types to native Python types (e.g., float32 to float)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

def cleanup_previous_emails(previous_emails_data, days_threshold=30):
    """Remove emails older than the specified threshold (in days)."""
    cutoff_date = time.time() - (days_threshold * 86400)  # Convert days to seconds
    return [
        entry for entry in previous_emails_data
        if time.mktime(time.strptime(entry['added_on'], "%Y-%m-%d %H:%M:%S")) >= cutoff_date
    ]
# ============================
# 4. MAIN EXECUTION
# ============================
if __name__ == "__main__":
    logging.info("Starting enhanced email classification pipeline")
    
    # Test files - automatically filter to existing files
    test_files = [f for f in [
        "Testing.eml", "TestingV1.eml"
    ] if os.path.exists(f)]
    
    if not test_files:
        logging.error("No test files found!")
        exit(1)
    
    previous_emails_data = load_previous_emails()
    previous_emails = [entry['text'] for entry in previous_emails_data]
    
    
    all_results = []
    for file_path in test_files:
        print(f"\nProcessing: {file_path}")
        results = process_email(file_path, previous_emails)
        
        if results:
            all_results.extend(results)
            for result in results:
                result = convert_numpy_types(result)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("=" * 60)          
            # Add processed emails to previous_emails
            previous_emails.extend([r['text'] for r in results])
            # Add timestamped entries to previous_emails_data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            previous_emails_data.extend([{'text': r['text'], 'added_on': timestamp} for r in results])
    
    all_results = convert_numpy_types(all_results)
    # Save comprehensive results
    with open('email_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save updated previous emails
    save_previous_emails(previous_emails_data)
    
    logging.info("Pipeline completed successfully. Results saved to email_results.json")