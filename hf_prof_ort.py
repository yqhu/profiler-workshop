from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import onnxruntime as ort
import torch
import time

# export onnx model before running it:
# python -m transformers.onnx --model=distilbert-base-uncased-finetuned-sst-2-english onnx/distilbert-base-uncased-finetuned-sst-2-english/

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ["I've been waiting for a HuggingFace course my whole life.",]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="np")

ort_session = ort.InferenceSession('onnx/distilbert-base-uncased-finetuned-sst-2-english/model.onnx')

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-prof-ort'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    
    outputs = ort_session.run(["last_hidden_state"], dict(inputs))
