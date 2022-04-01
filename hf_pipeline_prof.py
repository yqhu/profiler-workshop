# from the Hugging Face course: https://huggingface.co/course/
from transformers import pipeline
import torch

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            # on_trace_ready=torch.profiler.tensorboard_trace_handler('hf_pipeline_prof'),
                            with_stack=True, # needed for flamegraph viz
                            record_shapes=True) as prof:
    output = classifier("I've been waiting for a HuggingFace course my whole life.")

# print basic stats
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# export chrome trace
prof.export_chrome_trace("profiler_trace.json")

# export stacks for flamegraph visualization
prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

# use Brendan Gregg's FlameGraph tool to generate flamegraph/flamechart
# git clone https://github.com/brendangregg/FlameGraph
# ../FlameGraph/flamegraph.pl --title "FlameGraph" --countname "us." profiler_stacks.txt > perf_viz.svg
# ../FlameGraph/flamegraph.pl --title "FlameChart" --countname "us." --flamechart profiler_stacks.txt > perf_chart.svg
