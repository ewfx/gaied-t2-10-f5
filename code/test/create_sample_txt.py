import random
from datetime import datetime


def generate_email():
    if random.random() > 0.5:  # 50% chance for each type
        # Money movement email
        return f"""Subject: Transfer Request {datetime.now().strftime('%d-%m-%Y')}

Dear Team,

Please process this payment:

Amount: ${random.randint(1,20)},{random.randint(100,999):03d}
Account: ACCT-{random.randint(10000,99999)}
Date: {datetime.now().strftime('%d%b%y').upper()}
Reference: {random.choice(['Loan', 'Invoice', 'Settlement'])} Payment

Regards,
Client Services
"""
    else:
        # Adjustment email
        return f"""Subject: Rate Adjustment Request

Dear Loan Team,

Requesting rate modification:

Loan ID: LN-{random.randint(100000,999999)}
Current Rate: {random.uniform(3.0,8.0):.2f}%
Requested Rate: {random.uniform(2.0,7.5):.2f}%
Effective Date: {datetime.now().strftime('%Y-%m-%d')}

Thank you,
Account Management
"""

with open("sample_email.txt", "w") as f:
    f.write(generate_email())
print("Generated sample_email.txt")