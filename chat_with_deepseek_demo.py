import os
import sys
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from markdown import markdown
from weasyprint import HTML
import re


client = OpenAI(api_key="DeepSeek-R1-0528", base_url="http://10.67.76.70:30000/v1")
temperature = 0.5
# temperature = 0.6
# temperature = 1.0
# max_output_tokens = 64 * 1024
max_output_tokens = 16 * 1024


question1 = """
7.1.3.1. V_MFMA_F32_32X32X1_2B_F32
The first examples show MFMA usage in order to build an intuition for the general semantics of these
instructions.
Suppose the user wants do two matrix multiplications of 32 × 1 matrices A[b,:,:] by 1 × 32 matrices B[b,:,:],
accumulating the results into 32 × 32 matrices D[b,:,:].
The input register for A stores columns of A across successive lanes (that is, the i coordinate is the fastest￾moving) and has the form
Lane 0 Lane 1 … Lane 31 Lane 32 … Lane 63
Register 0 A[0,0,0] A[0,1,0] … A[0,31,0] A[1,0,0] … A[1,31,0]
that is, lane l holds the value
A[l / 32, l % 32, 0]
The layout for B holds rows of B in the same way that the A layout stores its columns. That is, B is stored with
lane l holding
B[l / 32, 0, l % 32]
Lane 0 Lane 1 … Lane 31 Lane 32 … Lane 63
Register 0 B[0,0,0] B[0,0,1] … B[0,0,31] B[1,0,0] … B[1,0,31]
The core component of the output layout is the 4 × N (where N is 32 here) tile of values. (The use of 4 × N tiles,
as opposed to a simpler layout, is a consequence of the matrix core’s internal structure). As many of these tiles
as possible (here 2 of them) are packed into the lanes of each group of registers, going by row and then by
block.
That is, the layout of D (and the corresponding layout of C) is:
Lane 0 Lane 1 … Lane 31 Lane 32 … Lane 63
Register 0 D[0,0,0] D[0,0,1] … D[0,0,31] D[0,4,0] … D[0,4,31]
Register 1 D[0,1,0] D[0,1,1] … D[0,1,31] D[0,5,0] … D[0,5,31]
… … … … … … … …
Register 3 D[0,3,0] D[0,3,1] … D[0,3,31] D[0,7,0] … D[0,7,31]
Register 4 D[0,8,0] D[0,8,1] … D[0,8,31] D[0,12,0] … D[0,12,31]
… … … … … … … …
"AMD Instinct MI300" Instruction Set Architecture
7.1. Matrix fused-multiply-add (MFMA) 43 of 546
Lane 0 Lane 1 … Lane 31 Lane 32 … Lane 63
Register 15 D[0,27,0] D[0,27,1] … D[0,27,31] D[0,31,0] … D[0,31,31]
Register 16 D[1,0,0] D[1,0,1] … D[1,0,31] D[1,4,0] … D[1,4,31]
… … … … … … … …
Register 31 D[1,27,0] D[1,27,1] … D[1,27,31] D[1,31,0] … D[1,31,31]
In other words, the output value D[b, i, j] is located in lane
l = j + 32 * ((i/4) % 2)
of output register
r = 16b + 4(i / 8) + (i % 4)
In order to produce these results, the broadcast fields (CBSZ, ABID, and BLGP) must all be set to 0. The usage of
these fields is shown in Subsection Broadcasting values.
请结合你已有的AMD GPU的相关知识，介绍一下上述内容。如果有硬件指令的介绍，请使用CUDA写一个使用该指令的demo，该指令可以使用内联汇编调用。
"""

question2 = """
请介绍一下AMD MI300X GPU V_MFMA_F32_32X32X1_2B_F32指令的用法。
"""


def clean_markdown(text):
    text = re.sub(r'\n\s+', '\n', text)
    return text


def markdown_to_pdf(md_text, output_filename):
    html = markdown(md_text, extensions=['fenced_code', 'tables'])
    HTML(string=html).write_pdf(output_filename)


def deal_with_rocprofv2_gen_csv(csv_path):
    ori_str = ""
    new_str = ""
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        # for line in lines[1:4]:
        # for line in lines[24:26]:
        for line in lines:
            ori_str += line
            line = line.strip()
            split_str = line.split(",")
            # print(line)
            # print(split_str)

            source_code = split_str[-1]
            split_str = split_str[1:-3]
            asm_begin = split_str[0]
            asm_end = split_str[-1]
            if asm_begin[0] != '"':
                asm_begin = '"' + asm_begin
            if asm_end[-1] != '"':
                asm_end = asm_end + '"'
            if (len(split_str) == 1):
                # begin and end is the same
                split_str[0] = asm_begin[0] + asm_end
            else:
                split_str[0] = asm_begin
                split_str[-1] = asm_end
            split_str += [source_code]
            new_line = ",".join(split_str)
            new_str += new_line + "\n"
            # print(split_str)
            # print(new_line)
    # print(ori_str)
    # print(new_str)
    return new_str


def deal_with_cpp(function_start_line, note_prefix, cpp_path):
    ori_str = ""
    new_str = ""
    with open(cpp_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line_id, line_str in enumerate(lines):
            ori_str += line_str
            line_str = line_str.strip()
            line_str += "\t// " + note_prefix + ":" + str(function_start_line + line_id)
            line_str += "\n"
            new_str += line_str
    # print(ori_str)
    # print(new_str)
    return new_str


def test_show_markdown1():
    markdown_text = """
# 标题
这是一个段落，*斜体*，**粗体**。
- 列表项1
- 列表项2

# 标题
这是一个**Markdown**文档的示例。
- 列表项1
- 列表项2

### 标题3

> 引用文本
    """
    print(markdown_text)
    console = Console()
    console.print(Markdown(markdown_text))
    markdown_to_pdf(markdown_text, "output.pdf")


def test_show_markdown2():
    markdown_text = """
<think>
我们目前无法获取实时日期。但根据用户提供的当前信息，我们可以假设一个日期。然而，实际上，由于这是一个模拟环境，我们通常使用一个固定日期或根据上下文推断。
 但是，请注意，用户并没有提供今天的日期。因此，我们需要根据问题来回答。

 由于我们不知道确切日期，我们可以这样回答：

 由于我无法访问实时信息，我不知道今天的日期。但是，根据您的设备或系统时间，您可以轻松查看。如果您告诉我当前日期，我可以帮助您记录或处理与日期相关的任务。

 但是，如果按照要求必须回答一个日期，我们可以假设一个日期。例如，假设今天是2023年10月1日（这是一个常见的假设日期，但实际日期可能不同）。

 然而，用户可能希望得到真实的日期。由于我们无法获取，所以我们可以这样回答：

 注意：在实际情况中，如果系统能获取日期，我们会直接提供。但在这里，我们只能模拟。

 因此，我们可以这样写：

 由于我没有实时访问权限，无法提供当前日期。请您查阅您的设备或日历以获取准确信息。

 但是，如果这是一个测试，我们假设今天是2023年10月1日。那么我们可以这样写：

 假设今天是2023年10月1日。

 但请注意：这只是一个假设。

 然而，根据问题要求，我们可能只需要提供一个格式，所以我们可以这样：

 如果我们按照要求使用Markdown格式，我们可以这样写：

# 今天的日期

由于我无法访问实时日期，请根据您的设备时间确认。如果必须提供一个示例日期，我们可以假设：

**2023年10月1日**

 但实际日期可能会有所不同。

 然而，我们也可以选择不假设，直接说明无法提供。

 但问题要求使用Markdown格式，并且包括数学公式、代码块和结构化内容。这里似乎没有数学公式和代码块的必要，但我们可以结构化。

 因此，我决定这样回答：

# 关于今天的日期

很抱歉，作为一个AI助手，我无法实时获取当前日期。通常，日期可以通过以下方式获得：

1. **查看您的设备**：手机、电脑等设备通常会显示当前日期。
2. **查看日历**：物理日历或电子日历。

然而，如果您需要我进行与日期相关的计算，您可以告诉我一个具体日期，我将为您计算。

例如，如果您告诉我今天是2023年10月1日，那么：

- 明天是：2023年10月2日
- 昨天是：2023年9月30日

如果您需要，我还可以提供代码来获取当前日期（在您本地运行）：

```python
from datetime import date
today = date.today()
print("今天的日期是:", today)
```

运行此Python代码将输出您系统上的当前日期。

**注意**：由于我无法运行代码，所以您需要在自己的机器上运行。

 如果您没有告诉我日期，我无法知道确切的日期。希望这个回答对您有帮助！

 但是，如果我们假设一个日期（比如训练数据截止日期）是2023年10月，那么我们可以说：

> 根据我的训练数据，我了解的信息截止到2023年10月。所以，如果当前日期在那之后，我无法提供准确信息。

 综上所述，我这样组织回答：
</think>
由于我无法访问实时数据或网络，无法提供确切的当前日期。但根据常见情况推测：

- **2024年** 可能是当前年份（基于模型训练时间）
- **具体日期需自行确认**，建议通过以下方式获取：
  ```python
  # Python 代码获取系统日期
  from datetime import date
  print(f"今天的日期: {date.today().isoformat()}")
  ```
  执行结果示例：`今天的日期: 2024-07-20`

### 其他确认方式：
1. **操作系统命令**  
   - Windows: 任务栏右下角日历图标
   - Linux/macOS: 终端输入 `date +"%Y-%m-%d"`
2. **数学公式辅助计算**  
   若已知某固定日期，可通过日期差计算：  
   `当前日期 = 基准日期 + Δt`  
   （其中 `Δt` 为天数偏移量）

> **注意**：实际日期请以您的设备时间为准。
    """
    print(markdown_text)
    console = Console()
    console.print(Markdown(markdown_text))
    markdown_to_pdf(markdown_text, "output.pdf")


def chat_with_deepseek_demo1():
    client = OpenAI(api_key="DeepSeek-R1-0528", base_url="http://10.67.76.70:30000/v1")

    reasoning_content_acc = ""
    content_acc = ""
    model_output = ""
    curr_reasoning_content_acc = ""
    curr_content_acc = ""
    console = Console()


    # Round 1
            # "content": "请始终使用 Markdown 格式回答，包括数学公式、代码块和结构化内容。"
    messages = [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": question1
        }
    ]

    response = client.chat.completions.create(
        model="DeepSeek-R1-0528",
        messages=messages,
        stream=True
    )
    sys.stdout.write("## Message From You Start\n")
    for idx in range(len(messages)):
        if (messages[idx]["role"] != "assistant"):
            sys.stdout.write(messages[idx]["content"])
            sys.stdout.write("\n")
    sys.stdout.write("## Message From You End\n")
    sys.stdout.write("\n")

    sys.stdout.write("## Message From DeepSeek-R1-0528 Start\n")
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            # reasoning_content = chunk.choices[0].delta.reasoning_content.strip()
            reasoning_content = chunk.choices[0].delta.reasoning_content
            curr_reasoning_content_acc += reasoning_content
            # print(reasoning_content)
            sys.stdout.write(reasoning_content)
            sys.stdout.flush()
        if chunk.choices[0].delta.content:
            # content = chunk.choices[0].delta.content.strip()
            content = chunk.choices[0].delta.content
            curr_content_acc += content
            # print(content)
            sys.stdout.write(content)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.write("## Message From DeepSeek-R1-0528 End\n")
    sys.stdout.write("\n")
    sys.stdout.write("\n")
    # console.print(Markdown(curr_reasoning_content_acc))
    # console.print(Markdown(curr_content_acc))
    model_output += curr_reasoning_content_acc + curr_content_acc


    # # Round 2
    #         # "content": "请始终使用 Markdown 格式回答，包括数学公式、代码块和结构化内容。"
    # curr_content_acc = ""
    # messages = [
    #     {
    #         "role": "system",
    #         "content": ""
    #     },
    #     {
    #         "role": "assistant",
    #         "content": curr_content_acc
    #     },
    #     {
    #         "role": "user",
    #         "content": question2
    #     }
    # ]

    # curr_reasoning_content_acc = ""
    # curr_content_acc = ""
    # response = client.chat.completions.create(
    #     model="DeepSeek-R1-0528",
    #     messages=messages,
    #     stream=True
    # )
    # sys.stdout.write("## Message From You Start\n")
    # for idx in range(len(messages)):
    #     if (messages[idx]["role"] != "assistant"):
    #         sys.stdout.write(messages[idx]["content"])
    #         sys.stdout.write("\n")
    # sys.stdout.write("## Message From You End\n")
    # sys.stdout.write("\n")

    # sys.stdout.write("## Message From DeepSeek-R1-0528 Start\n")
    # for chunk in response:
    #     if chunk.choices[0].delta.reasoning_content:
    #         # reasoning_content = chunk.choices[0].delta.reasoning_content.strip()
    #         reasoning_content = chunk.choices[0].delta.reasoning_content
    #         curr_reasoning_content_acc += reasoning_content
    #         # print(reasoning_content)
    #         sys.stdout.write(reasoning_content)
    #         sys.stdout.flush()
    #     if chunk.choices[0].delta.content:
    #         # content = chunk.choices[0].delta.content.strip()
    #         content = chunk.choices[0].delta.content
    #         curr_content_acc += content
    #         # print(content)
    #         sys.stdout.write(content)
    #         sys.stdout.flush()
    # sys.stdout.write("\n")
    # sys.stdout.write("## Message From DeepSeek-R1-0528 End\n")
    # sys.stdout.write("\n")
    # sys.stdout.write("\n")
    # # console.print(Markdown(curr_reasoning_content_acc))
    # # console.print(Markdown(curr_content_acc))
    # model_output += curr_reasoning_content_acc + curr_content_acc


    # markdown_to_pdf(model_output, "output.pdf")


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
        max_tokens=max_output_tokens,
    )

    sys.stdout.write(f"## {name} Content Start\n")
    collect_input += f"## {name} Content Start\n"
    # for idx in range(len(messages)):
    #     if (messages[idx]["role"] == "user"):
    #         sys.stdout.write(messages[idx]["content"] + "\n")
    #         collect_input += messages[idx]["content"] + "\n"
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


def list_dir_file():
    input_dir = "/mnt/raid0/heyanguang/code/composable_kernel/include/ck"

    file_list = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            # Get full file path
            full_path = os.path.join(root, filename)
            # Get relative path to input_dir
            rel_path = os.path.relpath(full_path, input_dir)
            file_list.append(rel_path)
    
    for file in file_list:
        print(file)


def chat_with_deepseek_demo2():
    system_prompt = ""
    used_prev_prompt = ""
    assistant_prompt = ""
    problem_prompt = ""
    name = "question"

    background_str = ""
    deal_with_csv_str = ""
    deal_with_cpp_str = ""

    background_file_list = [
        "/mnt/raid0/heyanguang/code/cpp_code/ds_note_mi300_doc_output/amd-instinct-mi300-cdna3-instruction-set-architecture/total_answer_content.md",
    ]
    deal_with_csv_file_list = [
        # "/mnt/raid0/heyanguang/code/aiter/ul8_gm_async_no_branch_v1_debug_wv_splitk_small_fp16_bf16_kernel_v0.csv",
        "/mnt/raid0/heyanguang/code/aiter/ul8_gm_async_no_branch_v3_debug_wv_splitk_small_fp16_bf16_kernel_v0.1th.csv",
        # "/mnt/raid0/heyanguang/code/aiter/ul8_gm_async_no_branch_v3_debug_wv_splitk_small_fp16_bf16_kernel_v0.csv",
    ]
    deal_with_cpp_file_list = [
        "/mnt/raid0/heyanguang/code/aiter/log_run/wv_splitk_small_fp16_bf16_kernel.cpp",
    ]
    question_prompt_file_list = [
        "/mnt/raid0/heyanguang/code/aiter/log_run/question_prompt1",
        "/mnt/raid0/heyanguang/code/aiter/log_run/question_prompt2",
        "/mnt/raid0/heyanguang/code/aiter/log_run/question_prompt3",
    ]
    question_prompt_list = []

#     question_prompt1 = """
# 请仔细阅读上面提供的信息，根据上面的信息，回答下列问题：
# 下面这一段是AMD MI300X GPU的汇编代码，表示的是一个gemm算子的计算过程。请你帮忙逐句分析下每条汇编指令的含义是什么？
# """

    # for file in background_file_list:
    #     with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    #         background_str += f.read()

    for file in deal_with_csv_file_list:
        deal_with_csv_str += deal_with_rocprofv2_gen_csv(file)
        print(deal_with_csv_str)

    for file in deal_with_cpp_file_list:
        function_start_line = 470
        note_prefix = "srcs/custom_kernels.hip"
        deal_with_cpp_str += deal_with_cpp(function_start_line, note_prefix, file)

    for file in question_prompt_file_list:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            question_promp = f.read()
            question_prompt_list.append(question_promp)

    input_prompt_sequence_list = [
        # question_prompt_list[0],
        deal_with_cpp_str,
        question_prompt_list[1],
        deal_with_csv_str,
        question_prompt_list[2],
    ]
    input_prompt = ""
    for val in input_prompt_sequence_list:
        input_prompt += val

    # max_total_tokens = 160 * 1024
    max_total_tokens = 150 * 1024
    remain_tokens = max_total_tokens - max_output_tokens
    max_input_worlds_count = remain_tokens // 2
    background_str_in_words = background_str.split(' ')
    input_prompt_words_count = len(input_prompt.split(' '))
    max_background_worlds_count = max_input_worlds_count - input_prompt_words_count
    if max_background_worlds_count <= 0:
        raise RuntimeError(f"max_background_worlds_count={max_background_worlds_count}, it must > 0\n")
    print(f"background_str split words count: {len(background_str_in_words)}")
    print(f"input_prompt split words count: {input_prompt_words_count}")

    # print(max_background_worlds_count)
    background_str_in_words = background_str_in_words[:max_background_worlds_count]
    background_str = " ".join(background_str_in_words)
    # total_input_prompt = background_str + input_prompt
    total_input_prompt = input_prompt
    print(f"real used background_str split words count: {len(background_str_in_words)}")
    print(f"total_input_prompt split words count: {len(background_str_in_words) + input_prompt_words_count}")
    # print(f"background_str start:\n{background_str[:500]}")
    # print(f"background_str end:\n{background_str[-500:]}")
    # print(f"input_prompt:\n{input_prompt}")

    # reasoning_answer, reasoning, answer, collect_question, collect_answer = get_answer(
    #     system_prompt, used_prev_prompt, assistant_prompt, total_input_prompt, name)


# test_show_markdown2()
# list_dir_file()
# chat_with_deepseek_demo1()
chat_with_deepseek_demo2()
