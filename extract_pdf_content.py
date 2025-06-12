import os
import subprocess
import pymupdf
import pdfplumber
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract


def extract_pdf_info(pdf_path):
    page_range = list(range(47, 55))
    # page_range = list(range(0, 10))

    doc = pymupdf.open(pdf_path)
    # toc = doc.get_toc()
    # print("目录结构:")
    # for entry in toc:
    #     level, title, page = entry[:3]
    #     print(f"{'  ' * (level-1)}• {title} (P{page})")

    # full_text = ""
    # for idx, page in enumerate(doc):
    #     if idx in page_range:
    #         page_text = page.get_text()
    #         print(page_text)
    #         print(f"len(page_text)={len(page_text)}")
    #         print(f"page_text_words={len(page_text.split(' '))}")

    # print("\n\n**************************\n\n")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        # print(dir(pdf))
        for idx, page in enumerate(pdf.pages):
            # if idx in page_range:
            if True:
                page_text = page.extract_text()
                full_text += page_text
        # print(full_text)
        print(f"len(full_text)={len(full_text)}")
        print(f"full_text_words={len(full_text.split(' '))}")

    # text = extract_text(pdf_path)
    # print(text)

    # output_dir = "./output_picture_dir"
    # os.makedirs(output_dir, exist_ok=True)
    # # images = convert_from_path(pdf_path, dpi=1024)
    # images = convert_from_path(pdf_path, dpi=300)
    # print(images[0])
    # print(type(images[0]))
    # text = ""
    # for idx, img in enumerate(images[:7]):
    #     img.save(os.path.join(output_dir, "image_" + str(idx) + ".png"))
    #     text += pytesseract.image_to_string(img) + "\n"
    # print(text)


def get_pdf_file_text(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            full_text += page_text
    return full_text


def test():
    pdf_file_path = "/mnt/raid0/heyanguang/code/cpp_code/input_doc_dir/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf"
    # pdf_file_path = "/mnt/raid0/heyanguang/code/cpp_code/Composable_Kernel_Training_sessions_02_compute_bound_case.pdf"
    # pdf_file_path = "/mnt/raid0/heyanguang/code/cpp_code/Composable_Kernel_Training_sessions_01_Introduction.pdf"
    data = extract_pdf_info(pdf_file_path)
    dirname = os.path.dirname(pdf_file_path)
    base_name = os.path.basename(pdf_file_path)
    base_name, _ = os.path.splitext(base_name)
    print(dirname)
    print(base_name)

    # result = subprocess.run(["pandoc",
    #                          "/mnt/raid0/heyanguang/code/cpp_code/output_doc_dir/amd-instinct-mi300-cdna3-instruction-set-architecture/answer_1_8.md",
    #                          "-o",
    #                          "/mnt/raid0/heyanguang/code/cpp_code/output_doc_dir/amd-instinct-mi300-cdna3-instruction-set-architecture/answer_1_8.pdf",
    #                          "--pdf-engine=weasyprint",
    #                          ], capture_output=True, text=True)
    # print(f"stdout: {result.stdout}")
    # print(f"stderr: {result.stderr}")
    # print(f"returncode: {result.returncode}")

    # input_dir = "/mnt/raid0/heyanguang/code/cpp_code/input_doc_dir"
    # ONE_QUESTION_WORD_LIMIT = 2000

    # file_list = []
    # # Recursively collect all file paths relative to input_dir
    # for root, _, files in os.walk(input_dir):
    #     for filename in files:
    #         # Get full file path
    #         full_path = os.path.join(root, filename)
    #         # Get relative path to input_dir
    #         rel_path = os.path.relpath(full_path, input_dir)
    #         file_list.append(rel_path)

    # for rel_path in file_list:
    #     input_path = os.path.join(input_dir, rel_path)
    #     input_content = get_pdf_file_text(input_path)
    #     input_content_in_words = input_content.split(' ')
    #     input_question_count = (len(input_content_in_words) + ONE_QUESTION_WORD_LIMIT - 1) / ONE_QUESTION_WORD_LIMIT
    #     print(f"input_path={input_path}")
    #     print(f"input_question_count={input_question_count}")
    #     print(f"len(input_content_in_words)={len(input_content_in_words)}")
    #     print(input_content[:10000])
    #     print(input_content[10000:])

test()

