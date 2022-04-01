import ctranslate2
import fastBPE
import torch

# from https://github.com/OpenNMT/CTranslate2
# to convert fairseq model to ctranslate2 mode:
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
# tar xf wmt16.en-de.joined-dict.transformer.tar.bz2
# ct2-fairseq-converter --model_path wmt16.en-de.joined-dict.transformer/model.pt \
#     --data_dir wmt16.en-de.joined-dict.transformer --quantization int8_float16 \
#     --output_dir ende_ctranslate2

def decode(x):
    return (x + " ").replace("@@ ", "").rstrip()–

bpe = fastBPE.fastBPE('wmt16.en-de.joined-dict.transformer/bpecodes', 'wmt16.en-de.joined-dict.transformer/dict.en.txt')
inputs = ["I've been waiting for a HuggingFace course my whole life."]

translator = ctranslate2.Translator("ende_ctranslate2/", device="cuda")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            schedule=torch.profiler.schedule(wait=1, warmup=10, active=3, repeat=1),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('ctran2-prof'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    for i in range(100):
        with torch.profiler.record_function("encode"):
            inp = bpe.apply(inputs)[0].split(' ')
        with torch.profiler.record_function("inference"):
            output = translator.translate_batch([inp])
        with torch.profiler.record_function("decode"):
            output = decode(" ".join(output[0].hypotheses[0]))
        prof.step()
