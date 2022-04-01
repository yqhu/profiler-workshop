# from the Hugging Face course: https://huggingface.co/course/
from transformers import pipeline
import torch

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-pipeline-prof'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    output = classifier("I've been waiting for a HuggingFace course my whole life.")
