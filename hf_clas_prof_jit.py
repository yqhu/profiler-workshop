from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import time

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ["I've been waiting for a HuggingFace course my whole life.",]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torchscript=True)

model = torch.jit.trace(model, (inputs['input_ids'], inputs['attention_mask']))
# model = torch.jit.optimize_for_inference(model.eval())

device = torch.device('cuda')
model = model.to(device)
inputs = inputs.to(device)

# warm up
with torch.no_grad():
    for i in range(100):
        output = model(**inputs)

# benchmark no_grad()
start = time.perf_counter()
with torch.no_grad():
    for i in range(100):
        output = model(**inputs)
print(f'no_grad():, {1000 * (time.perf_counter() - start) / 100:.1f} ms')

# warm up
with torch.inference_mode():
    for i in range(100):
        output = model(**inputs)

# benchmark inference_mode()
start = time.perf_counter()
with torch.inference_mode():
    for i in range(100):
        output = model(**inputs)
print(f'inference_mode():, {1000 * (time.perf_counter() - start) / 100:.1f} ms')

# profile no_grad()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-jit-no-grad'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    with torch.no_grad():
        output = model(**inputs)

# profile inference_mode()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-jit-inference-mode'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    with torch.inference_mode():
        output = model(**inputs)

