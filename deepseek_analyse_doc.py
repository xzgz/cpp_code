import os
import sys
import re
import argparse
import subprocess
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from markdown import markdown
from weasyprint import HTML
import pdfplumber


client = OpenAI(api_key="DeepSeek-R1-0528", base_url="http://10.67.76.70:30000/v1")
temperature = 0.5
# temperature = 0.6
# temperature = 1.0
max_tokens = 64 * 1024
top_p = 1.0
stop = ["\n", "###"]
presence_penalty = 0.5
frequency_penalty = 0.5


# prev_prompt = """
# 请根据我提供给你的介绍AMD Instinct MI300 GPU指令集和硬件架构的文档，说一下你掌握的有关AMD Instinct MI300 GPU硬件方面的知识（尤其是MI300的指令集和硬件架构）。
# """
prev_prompt = """
请用简短精炼的语言总结一下我之前给你提供的介绍AMD Instinct MI300 GPU的指令集和硬件架构的文档都有哪些内容？
"""

# # system_prompt = ""
# system_prompt = """
# 你是一位精通AMD GPU（尤其是Instinct MI300 GPU）和NVIDIA GPU各方面硬件细节的硬件工程师，同时你也精通AMD GPU算子性能的优化。
# 我现在给你一些介绍AMD Instinct MI300 GPU的指令集和硬件架构的文档，需要你结合你之前已经掌握的有关AMD Instinct MI300 GPU硬件方面的知识，
# 以及你在AMD GPU上积累的算子优化的经验，向那些对AMD GPU硬件架构及AMD GPU算子优化方法不熟悉的软件工程师介绍这些文档。
# 对于一些重要的指令，请介绍它常见的应用场景，如何使用它来优化算子的性能，并提供如何使用这些指令的代码示例。
# 对于一些重要的硬件特性，请介绍它常见的应用场景，如何使用它来优化算子的性能，并提供如何使用这些指令的代码示例。
# 对于一些和NVIDIA GPU不太相同的硬件特性，请介绍它这样设计的好处以及可能存在的坏处。
# 所有的回答请用中文。
# """
system_prompt = """
你是一位对AMD GPU的指令集、硬件架构及算子优化方法不熟悉的软件工程师。但是以后需要做一些算子优化工作，所以需要熟悉相关内容。
我现在给你一些介绍AMD Instinct MI300 GPU的指令集和硬件架构的文档，需要你从中抽取出你不太熟悉的内容，然后用简短精炼的语言总结出来，
这样我下次让你去优化AMD GPU的算子时，你可以快速的查阅你自己总结的内容，然后针对AMD GPU的硬件特性将AMD GPU算子的性能优化到极致。
"""

question = ""
assistant_prompt = ""
problem_prompt = ""

CACHE_WORD_LIMIT = 64 * 1024  # 64K words
# CACHE_WORD_LIMIT = 110  # 64K words

ONE_QUESTION_WORD_LIMIT = 2000
ONE_FILE_MAX_QUESTION_COUNT = 8
# ONE_FILE_MAX_QUESTION_COUNT = 2
two_question_intersection_words = 400


def get_pdf_file_text(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            full_text += page_text
    return full_text


def save_str_to_md(str_content, md_path):
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(str_content)


def save_md_to_pdf(md_path, pdf_path):
    convert_to_pdf_result = subprocess.run(["pandoc",
                             "--pdf-engine=weasyprint",
                             md_path,
                             "-o",
                             pdf_path,
                             ], capture_output=True, text=True)
    print(f"convert_to_pdf_result.stdout: {convert_to_pdf_result.stdout}")
    print(f"convert_to_pdf_result.stderr: {convert_to_pdf_result.stderr}")
    print(f"convert_to_pdf_result.returncode: {convert_to_pdf_result.returncode}")


def get_answer(system_prompt_local, used_prev_prompt, assistant_prompt, problem_prompt, name):
    reasoning_content_acc = ""
    content_acc = ""
    model_output = ""

    messages = [
        {
            "role": "user",
            "content": used_prev_prompt
        },
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
    collect_input += f"## {name} Content Start\n"
    for idx in range(len(messages)):
        if (messages[idx]["role"] == "user"):
            sys.stdout.write(messages[idx]["content"] + "\n")
            collect_input += messages[idx]["content"] + "\n"
    sys.stdout.write(f"## {name} Content End\n\n")
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

    collect_output += curr_reasoning_content_acc
    collect_output += curr_content_acc
    sys.stdout.write(f"\n## {name} DeepSeek-R1-0528 Analysis End\n\n\n")
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

    # Process each file
    for rel_path in file_list:
        input_path = os.path.join(input_dir, rel_path)
        input_base_name = os.path.basename(input_path)
        input_base_name, _ = os.path.splitext(input_base_name)
        curr_file_output_path = os.path.join(output_dir, input_base_name)
        os.makedirs(curr_file_output_path, exist_ok=True)

        try:
            input_content = get_pdf_file_text(input_path)
            # input_content = input_content[40736:]
            input_content_in_words = input_content.split(' ')
            input_content_in_words_len = len(input_content_in_words)
            input_question_count = 1
            if input_content_in_words_len > ONE_QUESTION_WORD_LIMIT:
                tmp_len = input_content_in_words_len - ONE_QUESTION_WORD_LIMIT
                input_question_count += (tmp_len + ONE_QUESTION_WORD_LIMIT - two_question_intersection_words - 1) // (ONE_QUESTION_WORD_LIMIT - two_question_intersection_words)
            assistant_prompt = ""
            collect_output_one_file = ""
            total_collect_input_output_one_file = ""
            total_collect_output = ""
            total_collect_input_output = ""
            total_answer_content = ""
            print(f"input_path={input_path}")
            print(f"input_content_in_words_len={input_content_in_words_len}")
            print(f"input_question_count={input_question_count}")
            sys.stdout.flush()

            if input_question_count > 1:
                print("1111111111111111111111111111111111111111111")
                split_start_idx = 0
                question_count = 0
                used_prev_prompt = ""
                stop_words_len = input_content_in_words_len - two_question_intersection_words
                while split_start_idx < stop_words_len:
                    input_split_in_words = input_content_in_words[split_start_idx : split_start_idx + ONE_QUESTION_WORD_LIMIT]
                    assistant_prompt_in_words = assistant_prompt.split(' ')
                    if len(input_split_in_words) + len(assistant_prompt_in_words) > CACHE_WORD_LIMIT:
                        assistant_prompt_in_words = assistant_prompt_in_words[len(input_split_in_words) + len(assistant_prompt_in_words) - CACHE_WORD_LIMIT : ]
                    assistant_prompt = " ".join(assistant_prompt_in_words)
                    input_split = " ".join(input_split_in_words)
                    print(f"len(input_split_in_words)={len(input_split_in_words)}")
                    print(f"len(assistant_prompt_in_words)={len(assistant_prompt_in_words)}")
                    sys.stdout.flush()
                    if question_count != 0:
                        used_prev_prompt = prev_prompt

                    output_content_split, reasoning_content_split, answer_content_split, collect_input_split, collect_output_split = get_answer(
                        system_prompt, used_prev_prompt, assistant_prompt, input_split, name="question" + str(question_count + 1))

                    collect_output_one_file += collect_output_split
                    total_collect_input_output_one_file += collect_input_split
                    total_collect_input_output_one_file += collect_output_split
                    total_collect_output += collect_output_split
                    total_collect_input_output += collect_input_split
                    total_collect_input_output += collect_output_split
                    total_answer_content += answer_content_split
                    assistant_prompt += answer_content_split
                    assistant_prompt_in_words = assistant_prompt.split(' ')
                    print(f"len(assistant_prompt_in_words)={len(assistant_prompt_in_words)}")
                    sys.stdout.flush()
                    split_start_idx += ONE_QUESTION_WORD_LIMIT - two_question_intersection_words
                    question_count += 1

                    if (question_count % ONE_FILE_MAX_QUESTION_COUNT == 0) or (split_start_idx >= stop_words_len):
                        qst = question_count - ONE_FILE_MAX_QUESTION_COUNT + 1
                        qend = question_count

                        output_md_path = os.path.join(curr_file_output_path, "question_answer_" + str(qst) + "_" + str(qend) + ".md")
                        output_pdf_path = os.path.join(curr_file_output_path, "question_answer_" + str(qst) + "_" + str(qend) + ".pdf")
                        save_str_to_md(total_collect_input_output_one_file, output_md_path)
                        save_md_to_pdf(output_md_path, output_pdf_path)
                        output_md_path = os.path.join(curr_file_output_path, "answer_" + str(qst) + "_" + str(qend) + ".md")
                        output_pdf_path = os.path.join(curr_file_output_path, "answer_" + str(qst) + "_" + str(qend) + ".pdf")
                        save_str_to_md(collect_output_one_file, output_md_path)
                        save_md_to_pdf(output_md_path, output_pdf_path)

                        print(f"Processed: {question_count}/{input_question_count} questions of file {input_path}")
                        sys.stdout.write("\n")
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        collect_output_one_file = ""
                        total_collect_input_output_one_file = ""
            else:
                print("222222222222222222222222222222222222222222222222")
                used_prev_prompt = ""
                print(f"input_content_in_words_len={input_content_in_words_len}")
                sys.stdout.flush()
                output_content_split, reasoning_content_split, answer_content_split, collect_input_split, collect_output_split = get_answer(
                    system_prompt, used_prev_prompt, assistant_prompt, input_content, name="question1")
                collect_output_one_file += collect_output_split
                total_collect_input_output_one_file += collect_input_split
                total_collect_input_output_one_file += collect_output_split
                total_collect_output += collect_output_split
                total_collect_input_output += collect_input_split
                total_collect_input_output += collect_output_split
                total_answer_content += answer_content_split
                assistant_prompt += answer_content_split
                assistant_prompt_in_words = assistant_prompt.split(' ')
                print(f"len(assistant_prompt_in_words)={len(assistant_prompt_in_words)}")
                sys.stdout.flush()

                qst = 1
                qend = 1

                output_md_path = os.path.join(curr_file_output_path, "question_answer_" + str(qst) + "_" + str(qend) + ".md")
                output_pdf_path = os.path.join(curr_file_output_path, "question_answer_" + str(qst) + "_" + str(qend) + ".pdf")
                save_str_to_md(total_collect_input_output_one_file, output_md_path)
                save_md_to_pdf(output_md_path, output_pdf_path)
                output_md_path = os.path.join(curr_file_output_path, "answer_" + str(qst) + "_" + str(qend) + ".md")
                output_pdf_path = os.path.join(curr_file_output_path, "answer_" + str(qst) + "_" + str(qend) + ".pdf")
                save_str_to_md(collect_output_one_file, output_md_path)
                save_md_to_pdf(output_md_path, output_pdf_path)

                print(f"Processed: {1}/{1} questions of file {input_path}")
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                sys.stdout.flush()
                collect_output_one_file = ""
                total_collect_input_output_one_file = ""

            output_md_path = os.path.join(curr_file_output_path, "total_collect_output.md")
            output_pdf_path = os.path.join(curr_file_output_path, "total_collect_output.pdf")
            save_str_to_md(total_collect_output, output_md_path)
            save_md_to_pdf(output_md_path, output_pdf_path)
            print(f"Save total_collect_output to {output_md_path}")
            output_md_path = os.path.join(curr_file_output_path, "total_collect_input_output.md")
            output_pdf_path = os.path.join(curr_file_output_path, "total_collect_input_output.pdf")
            save_str_to_md(total_collect_input_output, output_md_path)
            save_md_to_pdf(output_md_path, output_pdf_path)
            print(f"Save total_collect_input_output to {output_md_path}")
            output_md_path = os.path.join(curr_file_output_path, "total_answer_content.md")
            output_pdf_path = os.path.join(curr_file_output_path, "total_answer_content.pdf")
            save_str_to_md(total_answer_content, output_md_path)
            save_md_to_pdf(output_md_path, output_pdf_path)
            print(f"Save total_answer_content to {output_md_path}")
            sys.stdout.write("\n")
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error processing {rel_path}: {str(e)}", file=sys.stderr)
            sys.stdout.flush()


def chat_with_deepseek():

    # # Set up command-line argument parser
    # parser = argparse.ArgumentParser(
    #     description="Process files: insert filename at start and save with normalized name"
    # )
    # parser.add_argument("input_dir", help="Source directory path")
    # parser.add_argument("output_dir", help="Target directory path")
    # args = parser.parse_args()

    input_dir = "/mnt/raid0/heyanguang/code/cpp_code/input_doc_dir"
    # output_dir = "/mnt/raid0/heyanguang/code/cpp_code/output_doc_en_dir"
    # output_dir = "/mnt/raid0/heyanguang/code/cpp_code/output_doc_dir"
    output_dir = "/mnt/raid0/heyanguang/code/cpp_code/ds_note_mi300_doc_output"
    process_files(input_dir, output_dir)


chat_with_deepseek()
