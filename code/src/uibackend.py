from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re
import logging

try:
    from email_classification import VALID_CATEGORIES, SUBTYPE_MAPPING, process_email
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from email_classification import VALID_CATEGORIES, SUBTYPE_MAPPING, process_email

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORY_IDS = {}

def generate_category_id():
    """Generates a unique ID for a new request type."""
    return max(CATEGORY_IDS.values(), default=0) + 1

@app.get("/categories")
def get_categories():
    """Returns available categories, mapping to subtypes if available."""
    result = [
        {
            "id": idx + 1,
            "requestType": category,
            "subRequestType": SUBTYPE_MAPPING.get(category, ["N/A"])
        }
        for idx, category in enumerate(VALID_CATEGORIES)
    ]
    return {"categories": result}

# Endpoint to process an email file
class FileName(BaseModel):
    file_path: str
 
@app.post("/process-email-file")
def process_email_file(file: FileName):
     """
     Process an email file by its file path using process_email()
     and return the result.
     """
     if not os.path.exists(file.file_path):
         raise HTTPException(status_code=404, detail=f"File {file.file_path} not found.")
     result = process_email(file.file_path)
     return {"result": result}
 
class RequestType(BaseModel):
    requestType: str
    subRequestType: List[str]

@app.post("/add-category")
def add_category(request: RequestType):
    """Adds a new request type and subrequest types."""
    existing_subrequests = SUBTYPE_MAPPING.get(request.requestType, [])

    if request.requestType in VALID_CATEGORIES and set(request.subRequestType) <= set(existing_subrequests):
        raise HTTPException(status_code=400, detail="This request type with the same sub-request types already exists.")

    if request.requestType not in CATEGORY_IDS:
        CATEGORY_IDS[request.requestType] = generate_category_id()

    if request.requestType in VALID_CATEGORIES:
        SUBTYPE_MAPPING[request.requestType] = list(set(existing_subrequests + request.subRequestType))
    else:
        VALID_CATEGORIES.append(request.requestType) 
        SUBTYPE_MAPPING[request.requestType] = request.subRequestType if request.subRequestType else ["N/A"]

    update_email_classification()

    for cat in VALID_CATEGORIES:
        if cat not in CATEGORY_IDS:
            CATEGORY_IDS[cat] = generate_category_id()

    return {
        "message": "Category added successfully",
        "id": CATEGORY_IDS[request.requestType],
        "categories": [
            {"id": CATEGORY_IDS[cat], "requestType": cat, "subRequestType": SUBTYPE_MAPPING.get(cat, ["N/A"])}
            for cat in VALID_CATEGORIES
        ],
    }

def update_email_classification():
    """Updates email_classification.py without erasing existing contents."""
    file_path = os.path.join(os.path.dirname(__file__), "email_classification.py")

    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        match_categories = re.search(r"VALID_CATEGORIES\s*=\s*(\{.*?\}|\[.*?\])", content, re.DOTALL)
        existing_categories = eval(match_categories.group(1)) if match_categories else set()
    
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return

    updated_categories = set(existing_categories).union(VALID_CATEGORIES)
    updated_subtypes = SUBTYPE_MAPPING.copy()

    updated_content = re.sub(
        r"VALID_CATEGORIES\s*=\s*(\{.*?\}|\[.*?\])",
        f"VALID_CATEGORIES = {sorted(updated_categories)}",
        content,
        flags=re.DOTALL,
    )

    updated_content = re.sub(
        r"SUBTYPE_MAPPING\s*=\s*(\{.*?\})",
        f"SUBTYPE_MAPPING = {updated_subtypes}",
        updated_content,
        flags=re.DOTALL,
    )

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(updated_content)
        print(f"✅ Successfully updated {file_path}")
    except Exception as e:
        print(f"❌ Error writing to {file_path}: {e}")

class DeleteSubRequestModel(BaseModel):
    requestId: int
    subRequest: str

@app.delete("/delete-sub-request")
def delete_sub_request(data: DeleteSubRequestModel):
    """Deletes a sub-request type from a given request type."""
    category_list = list(VALID_CATEGORIES)
    if data.requestId <= 0 or data.requestId > len(category_list):
        raise HTTPException(status_code=404, detail="Invalid request type ID")

    requestType = category_list[data.requestId - 1]

    if requestType not in SUBTYPE_MAPPING or data.subRequest not in SUBTYPE_MAPPING[requestType]:
        raise HTTPException(status_code=404, detail="Sub-request type not found")

    SUBTYPE_MAPPING[requestType].remove(data.subRequest)

    if not SUBTYPE_MAPPING[requestType]:
        del SUBTYPE_MAPPING[requestType]
        VALID_CATEGORIES.remove(requestType)

    update_email_classification()

    return {"message": "Sub-request deleted successfully"}

class DeleteRequest(BaseModel):
    requestId: int
    requestType: str
    subRequestType: List[str]

@app.delete("/delete-category")
def delete_category(request: DeleteRequest):
    """Deletes an entire request type along with its subtypes."""
    if request.requestType not in VALID_CATEGORIES:
        raise HTTPException(status_code=404, detail="Request type not found")

    # Remove the category and related subtypes
    VALID_CATEGORIES.remove(request.requestType)
    SUBTYPE_MAPPING.pop(request.requestType, None)
    CATEGORY_IDS.pop(request.requestType, None)

    update_email_classification()

    return {"message": "Request type deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
