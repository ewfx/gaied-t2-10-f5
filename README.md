# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
The project (GenAIEmailClassification) consists of multiple Python files (for example, the main email classification logic in email_classification.py and maintenance tasks in maintenance.py). It leverages several external libraries for tasks like text extraction, natural language processing, machine learning inference, duplicate detection, and scheduled maintenance.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
To provide user friendly solution ; which routes emails and provides bird eye view of customer request based on content. Helping users save precious time for customer service.

## ⚙️ What It Does
* User Interface to add and maintain new request and sub-request types.
* Auto training model (Trains based on LLM output) and trained model can be used (based on confidence) to avoid external calls to save cost.
* Local classification model in absence of external LLM.
* In depth email classification including attachment
* Provide hints for routing the email to destination team
* Duplicate recognition and hints on duplicate email (With description and importance based on intent of re-send email)


## 🛠️ How We Built It
Gen-Ai model built using python as back-end using:
Python, Spacy, Transformers (Hugging Face)
SentenceTransformers for embeddings
DeepSeek API for classification
pdfplumber, pytesseract for OCR
Logging, dotenv, requests for integrations


## 🚧 Challenges We Faced
Describe the major technical or non-technical challenges your team encountered.

## 🏃 How to Run
Dependencies
Below is a list of the external libraries used throughout the project along with their pip installation commands:

numpy – for numerical operations
pip install numpy

requests – for making HTTP requests to external APIs (e.g., DeepSeek)
pip install requests

pdfplumber – for extracting text from PDF files
pip install pdfplumber

python-docx – for extracting text from DOCX files
pip install python-docx

spacy – for natural language processing
pip install spacy

python-dotenv – for loading environment variables
pip install python-dotenv

sentence-transformers – for encoding text and computing embeddings
pip install sentence-transformers

transformers – for inference using Hugging Face models
pip install transformers

scikit-learn – for similarity metrics (e.g., cosine similarity)
pip install scikit-learn

schedule – for scheduling the cleanup maintenance job
pip install schedule

pytesseract – for OCR-based text extraction from images (used as fallback in text extraction)
pip install pytesseract

Pillow – for image processing (used with OCR)
pip install pillow


## Install the Dependencies:
Open your terminal, navigate to the project folder (e.g., on your Windows machine), and run:
pip install -r requirements.txt

Since the project uses spaCy for text processing, download the English language model by running:
python -m spacy download en_core_web_sm

## Add your API key in .env file
DEEPSEEK_API_KEY=<API Key>

   ```

## 🏗️ Tech Stack
- 🔹 Frontend: React
- 🔹 Backend:  Python / FastAPI
- 🔹 Database: -
- 🔹 Other: OpenAI API

Python, Spacy, Transformers (Hugging Face)
SentenceTransformers for embeddings
DeepSeek API for classification
pdfplumber, pytesseract for OCR
Logging, dotenv, requests for integrations


## 👥 Team
- Shravankumar Mudrebettu
- Atosh Veerabhadrannavar
- Rajendra Malya
- Kumara subramanya
