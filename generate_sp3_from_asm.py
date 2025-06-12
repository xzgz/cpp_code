import re
import argparse


def extract_kernel_assembly(file_path, kernel_name):
    """
    Extract specific kernel assembly code from a file based on the given kernel name.
    
    Args:
        file_path (str): Path to the assembly file
        kernel_name (str): Name of the kernel to extract
        
    Returns:
        str: Extracted assembly code, or empty string if not found
    """
    try:
        # Read assembly file content
        # with open(file_path, 'r') as f:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"

    # Build regex patterns for key markers
    start_pattern = re.escape(kernel_name) + r':\s*;\s*@' + re.escape(kernel_name)
    # start_pattern = kernel_name + r':\s*;\s*@' + kernel_name
    bb_pattern = r';\s*%bb\.0:'
    end_pattern = r'\s*s_endpgm'

    # Compile complete extraction pattern
    pattern = rf"""
        ({start_pattern})   # Match kernel declaration line
        (.*?)               # Non-greedy match for content to remove
        ({bb_pattern})      # Match basic block start
        (.*?)               # Non-greedy match for content to keep
        ({end_pattern})     # Match end instruction
    """

    # Create regex object with multiline matching
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

    # Search for pattern in file content
    match = regex.search(content)
    # print(match)
    if match:
        # Extract key components from match
        kernel_start = match.group(1)
        bb_start = match.group(3)
        preserved_code = match.group(4)
        end_instr = match.group(5)
        # Reconstruct the preserved assembly snippet
        return f"{kernel_start}\n{bb_start}{preserved_code}{end_instr}"

    return ""  # Return empty string if no match found


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract kernel assembly snippet.')
    parser.add_argument('--asm_file', required=True, help='Input assembly file')
    parser.add_argument('--kernel_txt', required=True, help='xxx_kernel.txt that include kernel_name')
    parser.add_argument('--out_sp3_file', required=True, help='xxx_kernel.txt that include kernel_name')
    args = parser.parse_args()
    asm_file = args.asm_file
    kernel_txt = args.kernel_txt
    out_sp3_file = args.out_sp3_file

    # asm_file = "/mnt/raid0/heyanguang/code/aiter/log_run/custom_kernels.cuda.ul8_gm_async_v3.s"
    # kernel_txt = "/mnt/raid0/heyanguang/code/aiter/ttv_dir_ul8_gm_async_v3/wv_splitk_small_fp16_bf16_kernel_v0_kernel.txt"

    kernel_name = ""
    try:
        with open(kernel_txt, 'r') as f:
            line1 = f.readlines()[0]
            kernel_name = line1.split(":")[-1].split(".")[0].strip()
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {asm_file}")
    # print(kernel_name)

    sp3_str_prefix = "shader main\nasic(GFX942)\n\n"
    sp3_str_suffix = "\n\nend\n"
    extracted_str = extract_kernel_assembly(asm_file, kernel_name)
    sp3_str = sp3_str_prefix + extracted_str + sp3_str_suffix
    # print(sp3_str)
    with open(out_sp3_file, 'w', encoding='utf-8') as f:
        f.write(sp3_str)
