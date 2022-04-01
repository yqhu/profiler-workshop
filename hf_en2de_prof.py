from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import torch
import time

mname = "allenai/wmt16-en-de-12-1"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

device = torch.device('cuda')
model = model.to(device)

inputs = ["I've been waiting for a HuggingFace course my whole life.",]
input_ids = tokenizer.encode(inputs, return_tensors="pt")
input_ids = input_ids.to(device)

# warm up
with torch.no_grad():
    for i in range(100):
        output = model.generate(input_ids)

# benchmark model.generate()
start = time.perf_counter()
with torch.no_grad():
    for i in range(100):
        output = model.generate(input_ids)
print(f'inference only: {1000 * (time.perf_counter() - start) / 100} ms')

# benchmark everything
start = time.perf_counter()
with torch.no_grad():
    for i in range(100):
        input_ids = tokenizer.encode(inputs, return_tensors="pt")
        input_ids = input_ids.to(device)
        output = model.generate(input_ids)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(f'tokenization + inference + decoding: {1000 * (time.perf_counter() - start) / 100} ms')

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            schedule=torch.profiler.schedule(wait=1, warmup=10, active=3, repeat=1),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-en2de-prof'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:

    with torch.no_grad():
        for i in range(100):
            with torch.profiler.record_function("tokenize"):
                input_ids = tokenizer.encode(inputs, return_tensors="pt")
                input_ids = input_ids.to(device)
            with torch.profiler.record_function("generate"):
                output = model.generate(input_ids)
            with torch.profiler.record_function("decode"):
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            prof.step()

