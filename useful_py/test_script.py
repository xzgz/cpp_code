import subprocess
import numpy as np
import torch
import hashlib


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


def show_all_torch_type():
    dtypes = {
        name: getattr(torch, name)
        for name in dir(torch)
        if isinstance(getattr(torch, name), torch.dtype)
    }

    for name, dtype in dtypes.items():
        print(f"{name}: {dtype}")
    for name, dtype in dtypes.items():
        if 'float8' in name:
            print(f"{name}: {dtype}, max: {torch.finfo(dtype).max}, min: {torch.finfo(dtype).min}")
    print(torch.half)
    print(torch.float16)
    print(torch.bfloat16)


def test_torch_tensor_and_python_list():
    arr = torch.arange(0, 20)
    # arr = arr.to("cuda")
    arr = arr.to("cuda:8")
    print(arr)
    print(arr.device)
    # arr = arr.to(torch.int64)
    arr = arr.to(torch.int32)
    print(arr.dtype)
    print(arr[1])
    print(arr[1].dtype)
    print(arr[1].item)
    print(arr[1].item())
    print(type(arr[1]))
    print(type(arr[1].item))
    print(type(arr[1].item()))
    print(arr[1].shape)
    print(arr[1].numel())

    arr_i = arr[5]
    arr_i_val = arr[5].item()
    print(arr[arr_i])
    print(arr[arr_i_val])

    idx_list = []
    for i in range(4, 9):
        idx = arr[i] - arr[1]
        idx = idx.item()
        idx_list.append(idx)
    idx_list_tensor = torch.tensor(idx_list)
    print(idx_list)
    print(idx_list_tensor)
    print(type(idx_list[1]))
    print(type(idx_list_tensor[1]))
    print(list(idx_list))
    print(list(idx_list_tensor))
    # print(arr[4, 5, 6]) # IndexError: too many indices for tensor of dimension 1
    print(arr[[4, 5, 6]])
    print(arr[[idx_list]])
    print(arr[[idx_list_tensor]])
    # x = 12L # SyntaxError: invalid decimal literal


def test_torch_create_tensor():
    def create_tensor(min, max, *args, **kwargs):
        print(args)
        print(kwargs)
        print(*args)
        print(1, 2, 3, 4)
        print(*kwargs)
        print('dtype', 'device')
        print(type(args))
        print(type(kwargs))
        print(type(2))
        # print(**kwargs) # TypeError: 'dtype' is an invalid keyword argument for print()
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    q_init_min, q_init_max = (-10, 10)
    batch_size, qo_len, head_num, head_dim = (3, 1024, 8, 128)
    shape = [batch_size * qo_len, head_num, head_dim]
    dtype = torch.float16
    device = "cuda:5"
    # device = 6
    q = create_tensor(
        q_init_min, q_init_max, shape, dtype=dtype, device=device)
        # q_init_min, q_init_max, batch_size * qo_len, head_num, head_dim, dtype=dtype, device=device)
        # q_init_min, q_init_max, batch_size * qo_len, head_num, head_dim, dtype=dtype).to(device)
    # print(q)
    # print(q.shape)
    # print(q.dtype)
    # print(q.device)
    # print(type(q.dtype))
    # print(type(q.device))

    ta = torch.randn(batch_size * qo_len, head_num, head_dim, dtype=dtype, device=device)
    ta = torch.randn(*shape, dtype=dtype, device=device)
    ta = torch.randn(shape, dtype=dtype, device=device)
    print(ta.shape)
    print(ta.device)

    dtype_str = 'torch.float16'
    dtype = eval(dtype_str)
    print(dtype)
    print(type(dtype))

    print(dtype_str.split('.'))
    dtype2 = getattr(torch, dtype_str.split('.')[1])
    print(dtype2)
    print(type(dtype2))

    arr = torch.arange(0, 20)
    print(arr[20:23])
    print(torch.cat([arr[10:15], arr[20:23]], dim=-1))


def test_store_load_torch_tensor():
    torch.manual_seed(137)
    size2type_map = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
    }

    def create_tensor(min, max, *args, **kwargs):
        print(args)
        print(kwargs)
        print(*args)
        print(1, 2, 3, 4)
        print(*kwargs)
        print('dtype', 'device')
        print(type(args))
        print(type(kwargs))
        # print(**kwargs) # TypeError: 'dtype' is an invalid keyword argument for print()
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    q_init_min, q_init_max = (-10, 10)
    batch_size, qo_len, head_num, head_dim = (3, 1024, 8, 128)
    shape = [batch_size * qo_len, head_num, head_dim]
    # dtype = torch.float8_e4m3fn
    # dtype = torch.float8_e4m3fnuz
    dtype = torch.float16
    device = "cuda:5"
    print(device)
    print(torch.device(device))
    print(type(device))
    print(type(torch.device(device)))

    # device = 6
    arr1 = create_tensor(
        q_init_min, q_init_max, shape, dtype=dtype, device=device)
    print(arr1.shape)
    print(arr1.dtype)
    print(arr1.reshape(-1)[:7])

    arr1_float8_e4m3fn = arr1.to(torch.float8_e4m3fn)
    print(arr1_float8_e4m3fn.reshape(-1)[:7])
    print(arr1_float8_e4m3fn.dtype)
    print(arr1_float8_e4m3fn.element_size())
    print(f"arr1_float8_e4m3fn.element_size()={arr1_float8_e4m3fn.element_size()}")
    arr1_uint8 = arr1_float8_e4m3fn.view(size2type_map[arr1_float8_e4m3fn.element_size()])
    print(arr1_uint8.reshape(-1)[:7])
    print(arr1_uint8.dtype)
    arr1_float8_e4m3fn_v2 = arr1_uint8.view(torch.float8_e4m3fn)
    print(arr1_float8_e4m3fn_v2.reshape(-1)[:7])
    print(arr1_float8_e4m3fn_v2.dtype)
    arr1_uint8_np = arr1_uint8.detach().cpu().numpy()
    print(arr1_uint8_np.reshape(-1)[:7])
    print(arr1_uint8_np.shape)
    print(arr1_uint8_np.dtype)
    np.save("./arr1_uint8_np.npy", arr1_uint8_np)
    arr1_uint8_np_ld = np.load("./arr1_uint8_np.npy")
    print(arr1_uint8_np_ld.reshape(-1)[:7])
    print(arr1_uint8_np_ld.shape)
    print(arr1_uint8_np_ld.dtype)
    arr1_float8_e4m3fn_v3 = torch.tensor(arr1_uint8_np_ld).to(device).view(torch.float8_e4m3fn)
    print(arr1_float8_e4m3fn_v3.reshape(-1)[:7])
    hash_value1 = hashlib.md5(arr1_float8_e4m3fn.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    hash_value2 = hashlib.md5(arr1_float8_e4m3fn_v3.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    print(f"arr1_float8_e4m3fn hash_value   ={hash_value1}")
    print(f"arr1_float8_e4m3fn_v3 hash_value={hash_value2}")

    arr1_bfloat16 = arr1.to(torch.bfloat16)
    print(arr1_bfloat16.reshape(-1)[:7])
    print(arr1_bfloat16.dtype)
    arr1_uint16 = arr1_bfloat16.view(size2type_map[arr1_bfloat16.element_size()])
    print(arr1_uint16.reshape(-1)[:7])
    print(arr1_uint16.dtype)
    arr1_bfloat16_v2 = arr1_uint16.view(torch.bfloat16)
    print(arr1_bfloat16_v2.reshape(-1)[:7])
    print(arr1_bfloat16_v2.dtype)
    arr1_uint16_np = arr1_uint16.detach().cpu().numpy()
    print(arr1_uint16_np.reshape(-1)[:7])
    print(arr1_uint16_np.shape)
    print(arr1_uint16_np.dtype)
    np.save("./arr1_uint16_np.npy", arr1_uint16_np)
    arr1_uint16_np_ld = np.load("./arr1_uint16_np.npy")
    print(arr1_uint16_np_ld.reshape(-1)[:7])
    print(arr1_uint16_np_ld.shape)
    print(arr1_uint16_np_ld.dtype)
    arr1_bfloat16_v3 = torch.tensor(arr1_uint16_np_ld).to(device).view(torch.bfloat16)
    print(arr1_bfloat16_v3.reshape(-1)[:7])
    hash_value1 = hashlib.md5(arr1_bfloat16.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    hash_value2 = hashlib.md5(arr1_bfloat16_v3.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    hash_value3 = hashlib.sha256(arr1_bfloat16_v3.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    print(f"arr1_bfloat16 hash_value            ={hash_value1}")
    print(f"arr1_bfloat16_v3 hash_value         ={hash_value2}")
    print(f"arr1_bfloat16_v3 sha256 hash_value  ={hash_value3}")
    print(f"type(hash_value1)={type(hash_value1)}")
    print(f"len(hash_value1)={len(hash_value1)}")
    print(f"len(hash_value3)={len(hash_value3)}")

    print(arr1.view(torch.int8).detach().cpu().numpy().dtype)
    print(arr1.view(torch.uint8).detach().cpu().numpy().dtype)
    print(arr1.view(torch.uint16).detach().cpu().numpy().dtype)
    print(arr1.view(torch.uint32).detach().cpu().numpy().dtype)
    print(arr1.view(torch.int).detach().cpu().numpy().dtype)
    print(arr1.view(torch.int4).detach().dtype)
    print(arr1.view(torch.int5).detach().dtype)
    print(arr1.view(torch.int6).detach().dtype)
    print(arr1.view(torch.int7).detach().dtype)
    # print(arr1.view(torch.int4).detach().cpu().numpy().dtype)
    # print(arr1.view(torch.int5).detach().cpu().numpy().dtype)

    print(arr1.ravel().shape)


# print_some_info()
# test_torch_no_contiguous_tensor()
# show_all_torch_type()
# test_torch_tensor_and_python_list()
# test_torch_create_tensor()
test_store_load_torch_tensor()

