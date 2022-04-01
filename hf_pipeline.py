# from the Hugging Face course: https://huggingface.co/course/
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

output = classifier("I've been waiting for a HuggingFace course my whole life.")

print('output', output)
