import subprocess
import numpy as np
import torch


def print_some_info():
    cpp_file = "/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build/ck/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async_hip.hpp"
    # modify_line_template = "    if(blockIdx.x + blockIdx.y + blockIdx.z + threadIdx.y + threadIdx.z == 0 && threadIdx.x == 0)"
    modify_line_template = "       threadIdx.x == 0)"
    focus_line_id = 43
    thread_idx_max = 256

    line_list = []
    with open(cpp_file, 'r', encoding='utf-8', errors='ignore') as f:
        line_list = f.readlines()
    # print(modify_line_template[:-2])
    all_thr_idx_list = list(range(thread_idx_max))
    all_thr_idx_list = np.array(all_thr_idx_list)
    # focus_thr_idx_list = all_thr_idx_list[[5]]
    focus_thr_idx_list = all_thr_idx_list
    print(focus_thr_idx_list)

    for thr_idx in focus_thr_idx_list:
        modify_line = modify_line_template[:-2] + str(thr_idx) + ")\n"
        line_list[focus_line_id] = modify_line

        modify_file_str = ""
        for line in line_list:
            modify_file_str += line
        # print(modify_line)
        # print(modify_file_str)

        with open(cpp_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(modify_file_str)

        run_shell_result = subprocess.run([
            "bash",
            "/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/run.sh"
            ], capture_output=True, text=True)
        print(f"run_shell_result.stdout:\n{run_shell_result.stdout}")
        print(f"run_shell_result.stderr:\n{run_shell_result.stderr}")
        print(f"run_shell_result.returncode: {run_shell_result.returncode}")


def test_torch_no_contiguous_tensor():
    tensorA = torch.arange(24).reshape(6, 4).to(torch.int32)
    tensorA = tensorA.to("cuda")
    print("tensorA.dtype:", tensorA.dtype)
    print("tensorA.device:", tensorA.device)
    print("tensorA.shape:", tensorA.shape)
    print("tensorA strides:", tensorA.stride())  # 如 (4,1)
    print("tensorA is contiguous:", tensorA.is_contiguous())  # True

    list_random = [3, 1, 5]
    # list_random = [3]
    # tensorB = tensorA[list_random]
    tensorB = tensorA[2:4]
    print("tensorB is contiguous:", tensorB.is_contiguous())  # False
    print("tensorB.device:", tensorB.device)
    print("tensorB.shape:", tensorB.shape)

    print("tensorB strides:", tensorB.stride())  # 如 (4,1)
    print("tensorB storage data_ptr:", tensorB.storage().data_ptr())
    print("tensorB         data_ptr:", tensorB.data_ptr())
    print("tensorA storage data_ptr:", tensorA.storage().data_ptr())
    print("tensorA         data_ptr:", tensorA.data_ptr())
    # 两个 storage 地址不同 → 独立存储


# print_some_info()
test_torch_no_contiguous_tensor()

