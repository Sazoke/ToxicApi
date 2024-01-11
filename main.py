import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.responses import JSONResponse

# load tokenizer and model weights
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

app = FastAPI()


@app.get('/toxic')
async def index(request: Request):
    body = await request.json()
    tokens = tokenizer.encode(body['text'], return_tensors='pt')

    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()
    result = {}
    for i in range(probabilities.shape[1]):
        result[model.config.id2label[i]] = float(probabilities[0][i])
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
