import time
import os
import logging
import schedule
from email_classification import cleanup_previous_emails, load_previous_emails, save_previous_emails

print("Current working directory:", os.getcwd())

# Configure logging for maintenance tasks
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="maintenance.log",
)

logging.info("Maintenance module started.")

def cleanup_job():
    logging.info("Starting cleanup job for previous_emails.json")
    file_path = "previous_emails.json"
    abs_path = os.path.abspath(file_path)
    logging.info(f"Using file path: {abs_path}")
    
    if os.path.exists(file_path):
        try:
            previous_emails_data = load_previous_emails(file_path)
            cleaned_data = cleanup_previous_emails(previous_emails_data, days_threshold=30)
            save_previous_emails(cleaned_data, file_path)
            logging.info("Cleanup job completed successfully.")
        except Exception as e:
            logging.error(f"Cleanup job failed: {str(e)}")
    else:
        logging.warning("previous_emails.json not found. Skipping cleanup job.")

def main():
    # Schedule the cleanup job to run once a day at 2:00 AM
    schedule.every().day.at("02:00").do(cleanup_job)
    logging.info("Maintenance scheduler started. Cleanup job scheduled at 02:00 daily.")
    
    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every minute

if __name__ == "__main__":
    main()