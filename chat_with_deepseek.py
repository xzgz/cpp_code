import os
import sys
import argparse
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from markdown import markdown
from weasyprint import HTML
import re


client = OpenAI(api_key="DeepSeek-R1-0528", base_url="http://10.67.76.70:30000/v1")
# temperature = 0.5
temperature = 0.6
# temperature = 1.0
max_tokens = 64 * 1024
top_p = 1.0
stop = ["\n", "###"]
presence_penalty = 0.5
frequency_penalty = 0.5

# 8.用尽量精炼的语言阐述上述内容。
# 现在给你一些composable_kernel代码仓的一些代码，请你仔细分析这些代码，重点阐述如下的问题：
# 1.主要有哪些函数和类？
# 2.这些函数和类的功能是什么？
# 3.这些函数和类用到了哪些C++的新特性？
# 4.这些函数和类用到了哪些C++程序的设计思想？
# 5.代码中用到了哪些优化技术？
# 6.代码中哪些优化手段是比较通用的优化手段，哪些优化手段是和AMD GPU的硬件架构相关的？
# 7.你也可以讲一些代码中你认为比较闪亮的点和你认为对系统地理解整个代码的逻辑比较重要的点。
# 你可以从多个角度介绍代码的逻辑和优化点，越详细越好。

# system_prompt = "请始终使用Markdown格式回答，包括数学公式、代码块和结构化内容。"
system_prompt = """
现在你是一位精通C++的资深程序员，尤其精通C++模板元编程。同时你还精通NVIDIA开源的cutlass和cute代码库。此外，你也精通AMD MI300X GPU的硬件架构和AMD MI300X GPU算子的优化，精通ROCm平台开源的composable_kernel代码库。
现在给你一段composable_kernel代码仓的代码，请你向一位对C++模板元编程、AMD GPU硬件架构及优化方法都很生疏的、只懂得一些C++基础知识的新手程序员分析和介绍这段代码，尽量能给他讲懂。
你可以从多个角度介绍代码的逻辑和优化点，越详细越好。
"""
# system_prompt = ""

question = """
\n详细分析下上面的代码实现了哪些功能？每个功能的具体逻辑是什么？\n
"""
assistant_prompt = ""
problem_prompt = ""

WORD_LIMIT = 64 * 1024  # 64K words
# WORD_LIMIT = 110  # 64K words


def get_answer(system_prompt_local, assistant_prompt, problem_prompt, name):
    reasoning_content_acc = ""
    content_acc = ""
    model_output = ""

    messages = [
        {
            "role": "system",
            "content": system_prompt_local
        },
        {
            "role": "assistant",
            "content": assistant_prompt
        },
        {
            "role": "user",
            "content": problem_prompt
        }
    ]

    curr_reasoning_content_acc = ""
    curr_content_acc = ""
    curr_total_content = ""
    collect_input = ""
    collect_output = ""
    response = client.chat.completions.create(
        model="DeepSeek-R1-0528",
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
        # top_p=top_p,
        # stop=stop,
        # presence_penalty=presence_penalty,
        # frequency_penalty=frequency_penalty

    sys.stdout.write(f"## {name} Content Start\n")
    sys.stdout.write("```cpp\n")
    collect_input += f"## {name} Content Start\n"
    collect_input += "```cpp\n"
    for idx in range(len(messages)):
        if (messages[idx]["role"] == "user"):
            sys.stdout.write(messages[idx]["content"])
            sys.stdout.write("\n")
            collect_input += messages[idx]["content"] + "\n"
    sys.stdout.write("```\n")
    sys.stdout.write(f"## {name} Content End\n")
    sys.stdout.write("\n")
    collect_input += "```\n"
    collect_input += f"## {name} Content End\n\n"

    sys.stdout.write(f"## {name} DeepSeek-R1-0528 Analysis Start\n")
    collect_output += f"## {name} DeepSeek-R1-0528 Analysis Start\n"
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_content = chunk.choices[0].delta.reasoning_content
            curr_reasoning_content_acc += reasoning_content
            sys.stdout.write(reasoning_content)
            sys.stdout.flush()
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            curr_content_acc += content
            sys.stdout.write(content)
            sys.stdout.flush()
        # if chunk.usage:
        #     prompt_tokens = chunk.usage["prompt_tokens"]
        #     completion_tokens = chunk.usage["completion_tokens"]
        #     sys.stdout.write(f"prompt_tokens={prompt_tokens}\n")
        #     sys.stdout.write(f"completion_tokens={completion_tokens}\n")
        #     sys.stdout.flush()
        # if chunk.choices and chunk.choices[0].finish_reason:
        #     finish_reason = chunk.choices[0].finish_reason
        #     sys.stdout.write(f"finish_reason={finish_reason}\n")
        #     sys.stdout.flush()
        # print(chunk)

    collect_output += curr_reasoning_content_acc
    collect_output += curr_content_acc
    sys.stdout.write("\n")
    sys.stdout.write(f"## {name} DeepSeek-R1-0528 Analysis End\n")
    sys.stdout.write("\n")
    sys.stdout.write("\n")
    collect_output += f"\n## {name} DeepSeek-R1-0528 Analysis End\n\n\n"
    curr_total_content += curr_reasoning_content_acc + curr_content_acc

    return (curr_total_content, curr_reasoning_content_acc, curr_content_acc, collect_input, collect_output)


def process_files(input_dir, output_dir):
    """
    Process all files in input directory: insert filename at beginning, 
    save to output directory with normalized filename
    
    Args:
        input_dir (str): Source directory path
        output_dir (str): Target directory path
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    file_list = []
    # Recursively collect all file paths relative to input_dir
    for root, _, files in os.walk(input_dir):
        for filename in files:
            # Get full file path
            full_path = os.path.join(root, filename)
            # Get relative path to input_dir
            rel_path = os.path.relpath(full_path, input_dir)
            file_list.append(rel_path)

    total_content_all_file = ""
    collect_output_all_file = ""
    assistant_prompt = ""
    # Process each file
    for rel_path in file_list:
        input_path = os.path.join(input_dir, rel_path)

        try:
            input_content = ""
            # Read file content with UTF-8 encoding
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                input_content = f.read()

            total_content_one_file = ""
            collect_output_one_file = ""
            # Prepend filename to content
            input_content = f"// FILE: {rel_path}\n\n{input_content}"
            input_content_words = len(input_content.split(' '))
            assistant_prompt_words = len(assistant_prompt.split(' '))
            total_prompt_words = input_content_words + assistant_prompt_words
            if total_prompt_words > WORD_LIMIT:
                truncation_words = total_prompt_words - WORD_LIMIT
                if truncation_words >= assistant_prompt_words:
                    collect_output = ""
                    total_content = ""
                    assistant_prompt = ""
                    split_start_idx = 0
                    print(f"input_content_words={input_content_words}")
                    while split_start_idx < input_content_words:
                        assistant_prompt_words = len(assistant_prompt.split(' '))
                        input_split = " ".join(input_content.split(' ')[split_start_idx : split_start_idx + WORD_LIMIT - assistant_prompt_words])
                        input_split_content_words = len(input_split.split(' '))
                        print("1111111111111111111111111111111111111111111")
                        print(f"input_split_content_words={input_split_content_words}")
                        print(f"assistant_prompt_words={assistant_prompt_words}")
                        sys.stdout.flush()
                        output_content_split, reasoning_content_split, final_content_split, collect_input_split, collect_output_split = get_answer(
                            system_prompt, assistant_prompt, input_split, name=rel_path)
                        assistant_prompt += final_content_split
                        collect_output += collect_output_split
                        total_content += collect_input_split
                        total_content += collect_output_split
                        assistant_prompt_words = len(assistant_prompt.split(' '))
                        print(f"assistant_prompt_words={assistant_prompt_words}")
                        sys.stdout.flush()
                        if assistant_prompt_words >= WORD_LIMIT:
                            assistant_prompt = ""
                            assistant_prompt_words = len(assistant_prompt.split(' '))
                        split_start_idx += WORD_LIMIT - assistant_prompt_words
                    collect_output_one_file = collect_output
                    total_content_one_file = total_content
                else:
                    used_assistant_prompt_words = WORD_LIMIT - input_content_words
                    assistant_prompt = " ".join(assistant_prompt.split(' ')[-used_assistant_prompt_words:])
                    assistant_prompt_words = len(assistant_prompt.split(' '))
                    print("222222222222222222222222222222222222222222222222")
                    print(f"input_content_words={input_content_words}")
                    print(f"assistant_prompt_words={assistant_prompt_words}")
                    sys.stdout.flush()
                    output_content, reasoning_content, final_content, collect_input, collect_output = get_answer(
                        system_prompt, assistant_prompt, input_content, name=rel_path)
                    assistant_prompt += final_content
                    collect_output_one_file = collect_output
                    total_content_one_file += collect_input
                    total_content_one_file += collect_output
                    print(f"assistant_prompt_words={len(assistant_prompt.split(' '))}")
                    sys.stdout.flush()
            else:
                assistant_prompt_words = len(assistant_prompt.split(' '))
                print("33333333333333333333333333333333333333333333333333")
                print(f"input_content_words={input_content_words}")
                print(f"assistant_prompt_words={assistant_prompt_words}")
                sys.stdout.flush()
                output_content, reasoning_content, final_content, collect_input, collect_output = get_answer(
                    system_prompt, assistant_prompt, input_content, name=rel_path)
                assistant_prompt += final_content
                collect_output_one_file = collect_output
                total_content_one_file += collect_input
                total_content_one_file += collect_output
                print(f"assistant_prompt_words={len(assistant_prompt.split(' '))}")
                sys.stdout.flush()

            collect_output_all_file += collect_output_one_file
            total_content_all_file += total_content_one_file

            # Generate output filename: replace path separators with underscores
            output_filename = rel_path.replace(os.sep, "_")
            output_base_name, _ = os.path.splitext(output_filename)
            question_answer_output_filename = output_base_name + ".question_answer.md"
            answer_output_filename = output_base_name + ".answer.md"

            output_path = os.path.join(output_dir, question_answer_output_filename)
            # Write modified content to new file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(total_content_one_file)
            print(f"Processed: {rel_path} -> {question_answer_output_filename}")
            sys.stdout.flush()

            output_path = os.path.join(output_dir, answer_output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(collect_output_one_file)
            print(f"Processed: {rel_path} -> {answer_output_filename}")
            sys.stdout.write("\n")
            sys.stdout.write("\n")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error processing {rel_path}: {str(e)}", file=sys.stderr)
            sys.stdout.flush()

    all_file_output_path = os.path.join(output_dir, "all_file_question_answer.md")
    with open(all_file_output_path, 'w', encoding='utf-8') as f:
        f.write(total_content_all_file)
    print(f"Processed: {input_dir} -> {all_file_output_path}")
    sys.stdout.flush()

    all_file_output_path = os.path.join(output_dir, "all_file_answer.md")
    with open(all_file_output_path, 'w', encoding='utf-8') as f:
        f.write(collect_output_all_file)
    print(f"Processed: {input_dir} -> {all_file_output_path}")
    sys.stdout.write("\n")
    sys.stdout.flush()


def chat_with_deepseek():

    # # Set up command-line argument parser
    # parser = argparse.ArgumentParser(
    #     description="Process files: insert filename at start and save with normalized name"
    # )
    # parser.add_argument("input_dir", help="Source directory path")
    # parser.add_argument("output_dir", help="Target directory path")
    # args = parser.parse_args()

    # input_dir = "/mnt/raid0/heyanguang/code/cpp_code/input_dir"
    input_dir = "/mnt/raid0/heyanguang/code/composable_kernel/include/ck"
    output_dir = "/mnt/raid0/heyanguang/code/cpp_code/output_dir"
    process_files(input_dir, output_dir)


chat_with_deepseek()
