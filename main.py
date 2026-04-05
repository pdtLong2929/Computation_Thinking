import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer

# 1. Khởi tạo ứng dụng FastAPI [cite: 37]
app = FastAPI(title="Text Summarization API - Phạm Đình Tiểu Long")

# 2. Cấu hình và tải mô hình BART trực tiếp để tránh lỗi Pipeline [cite: 15, 21]
MODEL_NAME = "facebook/bart-large-cnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Đang tải mô hình {MODEL_NAME} trên thiết bị: {device}...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print("Tải mô hình thành công!")

# 3. Định nghĩa cấu trúc dữ liệu đầu vào [cite: 42, 44]
class Document(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

# --- CÁC ENDPOINT THEO YÊU CẦU CỦA BÀI LAB ---

@app.get("/")
def home():
    return {
        "project": "Automatically Concise Summary System",
        "student": "Phạm Đình Tiểu Long",
        "model": "BART-large-CNN",
        "description": "API help to change long document to short document"
    }

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": str(device),
        "model_ready": True
    }

@app.post("/predict")
def summarize_text(doc: Document):
    
    if not doc.text.strip():
        raise HTTPException(status_code=400, detail="Document not blank")
    
    if len(doc.text) < 50:
        raise HTTPException(status_code=400, detail="Document too short (At least 50 chars)")

    try:
        inputs = tokenizer([doc.text], max_length=1024, return_tensors="pt", truncation=True).to(device)

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=doc.max_length,
            min_length=doc.min_length,
            early_stopping=True,
            forced_bos_token_id=0
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {
            "original_length": len(doc.text),
            "summary_text": summary_text,
            "summary_length": len(summary_text)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
