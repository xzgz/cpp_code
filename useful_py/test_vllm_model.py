import sys
import json
import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def show_pt_trace():
    # trace_file = "./vllm_profile/hjbog-srdc-24_29372.1752551815447666629.pt.trace.json"
    trace_file = "./vllm_profile/hjbog-srdc-24_32548.1752552505516323528.pt.trace.json"
    print(f"open file {trace_file}")
    with open(trace_file, "r") as f:
        trace_data = json.load(f)

    cpu_events = [event for event in trace_data["traceEvents"] 
                if event.get("cat") == "CPU" and "dur" in event]

    top_cpu_events = sorted(cpu_events, key=lambda x: x["dur"], reverse=True)[:10]
    for event in top_cpu_events:
        print(f"操作: {event['name']}, 耗时(us): {event['dur']}, 调用栈: {event.get('stackTrace', '无')}")


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    # Create an LLM.
    kv_cache_dtype = "auto"
    # kv_cache_dtype = "fp8"
    model_path = "/data/heyanguang/code/models/Llama-3.1-8B-Instruct-FP8-KV"

    llm = LLM(
        model=model_path,
        dtype=torch.float16,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        block_size=16,
        kv_cache_dtype=kv_cache_dtype,
        quantization="fp8",
        # compilation_config=CompilationConfig(full_cuda_graph=True),
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    with torch.profiler.profile(with_stack=True,             
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./vllm_profile")) as prof:
        outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    # import vllm
    # print(vllm)
    # print(vllm.__version__)
    # main()
    show_pt_trace()
