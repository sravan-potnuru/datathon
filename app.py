from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
from fastapi import FastAPI

app = FastAPI()

# Load the pre-trained SVM model and TF-IDF vectorizer
svm_classifier = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Create a templates object for rendering HTML
templates = Jinja2Templates(directory="templates")

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_item(request):
    # Render the HTML page
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_emotion(request_data: TextRequest):  # Update the parameter name here
    text = request_data.text  # Use request_data
    # Vectorize the input text
    text_vectorized = tfidf_vectorizer.transform([text])
    # Make predictions
    prediction = svm_classifier.predict(text_vectorized)
    return {"emotion": prediction[0]}
