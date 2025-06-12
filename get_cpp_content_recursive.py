import os
import re
from collections import deque


WORD_LIMIT = 30 * 1024  # 128K words


def collect_files(root_dir):
    """Recursively collect relative paths of all files under root_dir"""
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            # Get relative path and normalize to forward slashes
            rel_path = os.path.relpath(os.path.join(root, file), root_dir)
            file_list.append(rel_path.replace('\\', '/'))
    for file in file_list:
        print(file)
    return file_list

def extract_includes(content):
    """Extract included filenames from file content using regex"""
    pattern = r'#include\s*[<"](.*?)[>"]'  # Matches both <...> and "..." includes
    return re.findall(pattern, content)

def find_closest_match(matches, current_file):
    """Find the closest matching file based on directory relationship"""
    current_dir = os.path.dirname(current_file)
    
    # Group matches by directory distance
    dir_levels = {}
    for match in matches:
        match_dir = os.path.dirname(match)
        
        # Calculate directory distance
        if match_dir == current_dir:
            distance = 0  # Same directory (closest)
        elif current_dir.startswith(match_dir):
            distance = current_dir.count(os.sep) - match_dir.count(os.sep)  # Parent directory
        else:
            # Find common path prefix
            common = os.path.commonpath([current_dir, match_dir])
            up_steps = current_dir[len(common):].count(os.sep) if common else current_dir.count(os.sep)
            down_steps = match_dir[len(common):].count(os.sep) if common else match_dir.count(os.sep)
            distance = up_steps + down_steps + 10  # Add penalty for non-related paths
        
        dir_levels.setdefault(distance, []).append(match)
    
    # Get matches with smallest distance
    min_distance = min(dir_levels.keys())
    closest_matches = dir_levels[min_distance]
    
    # Return the first match in alphabetical order if multiple at same level
    return sorted(closest_matches)[0]

def main(root_dir, file_focus, output_file):
    # Step 1: Recursively scan directory to get file list
    file_list = collect_files(root_dir)

    # Step 2: Initialize queue with focus file and create output file
    queue = deque([file_focus])
    processed = set([file_focus])  # Track processed files to avoid cycles

    # Initialize word counter
    total_words = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        while queue:
            # Step 3: Process next file in queue
            rel_path = queue.popleft()
            abs_path = os.path.join(root_dir, rel_path)

            try:
                # Read file content
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")
                continue

            header_line = f"//**************************file {rel_path} start**************************\n"
            tail_line = f"//**************************file {rel_path} end**************************\n\n\n"
            full_content = header_line + content + tail_line
            full_content_words = len(full_content.split(' '))

            # Check word limit before writing
            if total_words + full_content_words > WORD_LIMIT:
                # Write partial content if possible
                if total_words < WORD_LIMIT:
                    # Calculate how many words we can still write
                    remaining_words = WORD_LIMIT - total_words
                    # Write only part of the content
                    words = full_content.split(' ')
                    partial_content = " ".join(words[:remaining_words])
                    outfile.write(partial_content + "\n")
                    total_words += len(partial_content.split(' '))

                print(f"Word limit reached: {total_words} words (limit: {WORD_LIMIT})")
                break

            # Write to output file
            outfile.write(full_content)
            total_words += full_content_words
            print(f"Added {rel_path} ({full_content_words} words), total: {total_words}/{WORD_LIMIT}")

            # Extract included filenames from content
            inc_list = extract_includes(content)

            # Step 4: Find matching files for each include
            for inc in inc_list:
                # print(f"inc={inc}")
                # Find files ending with include name that haven't been processed
                matches = [f for f in file_list 
                          if f.endswith(inc) and f not in processed]

                if not matches:
                    continue
                # Select the closest match based on directory relationship
                if len(matches) == 1:
                    match = matches[0]
                else:
                    match = find_closest_match(matches, rel_path)

                queue.append(match)
                processed.add(match)
                # print(f"Found match: {inc} -> {match}")

                # # Step 5: Add matches to queue for processing
                # for match in matches[:1]:
                #     queue.append(match)
                #     processed.add(match)
                #     print(f"Found match: {inc} -> {match}")

if __name__ == "__main__":
    # root_dir = input("Enter root directory path: ").strip()
    # file_focus = input("Enter relative path of focus file: ").strip()
    # root_dir = "/mnt/raid0/heyanguang/code/aiter"
    # file_focus = "/mnt/raid0/heyanguang/code/aiter/csrc/kernels/custom_kernels.cu"

    root_dir = "/mnt/raid0/heyanguang/code/composable_kernel"
    file_focus = "/mnt/raid0/heyanguang/code/composable_kernel/example/01_gemm/gemm_dl_fp16.cpp"

    output_file = "./focus_file.cpp"

    file_focus_relative_path = os.path.relpath(file_focus, root_dir)
    print(file_focus_relative_path)

    main(root_dir, file_focus_relative_path, output_file)
    print("Processing complete! Result saved in focus_file.cpp")
