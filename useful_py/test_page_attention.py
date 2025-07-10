import itertools
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import hashlib

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.rocm import use_rocm_custom_paged_attention

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


def load_function_inputs(dump_dir, skip_data_tensor_name_list):
    import os
    import json
    import numpy as np

    """
    Loads function inputs from the specified directory,
    fully restoring tensor layouts (including non-contiguous ones).
    """

    def create_tensor(min, max, *args, **kwargs):
        # print(args)
        # print(kwargs)
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    # Read metadata file
    metadata_path = os.path.join(dump_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    params = {}

    # Load non-tensor parameters
    for name, value in metadata['non_tensors'].items():
        # Special handling for window_size tuple
        params[name] = value

    # Load tensor parameters
    for name, tensor_meta in metadata['tensors'].items():
        shape = tensor_meta['shape']
        # dtype_str = tensor_meta['dtype'].split('.')[1]
        dtype_str = tensor_meta['dtype'].replace("torch.", "")
        dtype = getattr(torch, dtype_str)
        device = tensor_meta['device']
        contiguous_tensor = None

        if name in skip_data_tensor_name_list:
            print(f"tensor {name} create from scratch")
            # print(shape)
            # print(type(dtype))
            # print(type(device))
            init_min = None
            init_max = None
            if name in "kv":
                init_min, init_max = (-5, 5)
            else:
                init_min, init_max = (-10, 10)
            contiguous_tensor = create_tensor(
                init_min, init_max, shape, dtype=torch.float32, device=device)
            contiguous_tensor = contiguous_tensor.to(dtype)
            # if "cache" in name:
            #     contiguous_tensor = contiguous_tensor.to(torch.int8)
            # else:
            #     contiguous_tensor = contiguous_tensor.to(dtype)
        else:
            print(f"tensor {name} load data from disk")
            file_path = os.path.join(dump_dir, tensor_meta['file'])
            data = np.load(file_path)
            # Create contiguous PyTorch tensor from numpy data
            contiguous_tensor = torch.tensor(data).reshape(shape)
            # Restore original data type
            contiguous_tensor = contiguous_tensor.view(dtype)
            # Restore original device placement
            if tensor_meta['device'] != 'cpu':
                device = torch.device(tensor_meta['device'])
                contiguous_tensor = contiguous_tensor.to(device)

        params[name] = contiguous_tensor

    return params


def load_and_run_test():
    from vllm.attention.ops.rocm_aiter_paged_attn import (AITERPagedAttention)
    AITERPagedAttention.is_asm_supported = True
    paged_attn = AITERPagedAttention()

    root_data_dir = "/lab-mlperf-inference/code/op_paged_attention_rocm_input/llama2_70b/dk2_eager_step11"
    root_data_dir2 = "/lab-mlperf-inference/code/op_pa_fwd_asm_input/llama2_70b/dk2_eager_step11"
    # skip_data_tensor_name_list = ["tmp_out", "query", "key_cache", "value_cache"]
    skip_data_tensor_name_list = []
    select_sample_id = [1, 22]

    sample_count = 25
    # sample_count = 10
    for sample_id in range(0, sample_count):
        if sample_id not in select_sample_id:
            continue

        dump_dir = root_data_dir + "/" + str(sample_id) + "/"
        dump_dir2 = root_data_dir2 + "/" + str(sample_id) + "/"
        params = load_function_inputs(dump_dir, skip_data_tensor_name_list)
        params2 = load_function_inputs(dump_dir2, skip_data_tensor_name_list)

        print("Loaded parameters summary:")
        for name, value in params.items():
            if isinstance(value, torch.Tensor):
                print(f"{name}: tensor(shape={value.shape}, dtype={value.dtype}, device={value.device}, "
                    f"contiguous={value.is_contiguous()}, stride={value.stride()})")
            elif isinstance(value, tuple) or isinstance(value, list):
                print(f"{name}: {type(value).__name__} = {value}")
                print(f"value[0] type: {type(value[0]).__name__}")
            else:
                print(f"{name}: {type(value).__name__} = {value}")

        for name, value in params2.items():
            if isinstance(value, torch.Tensor):
                print(f"{name}: tensor(shape={value.shape}, dtype={value.dtype}, device={value.device}, "
                    f"contiguous={value.is_contiguous()}, stride={value.stride()})")
            elif isinstance(value, tuple) or isinstance(value, list):
                print(f"{name}: {type(value).__name__} = {value}")
                print(f"value[0] type: {type(value[0]).__name__}")
            else:
                print(f"{name}: {type(value).__name__} = {value}")

        print(params["block_tables"])
        print(params2["block_tables"])
        print(params["block_size"])
        print(params2["block_size"])
        print(params["seq_lens"])
        print(params2["seq_lens"])

        # print(params["query"].reshape(-1)[:10])
        print(params["key_cache"].reshape(-1)[:10])
        print(params["out"].reshape(-1)[:10])
        print(params["kv_cache_dtype"])
        print(params["k_scale"])
        print(params["fp8_out_scale"])

        hash_value = hashlib.md5(params["out"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        print(f"params['out'] hash_value={hash_value}")
        params["out"][:] = 0
        print(params["out"].reshape(-1)[:10])
        hash_value = hashlib.md5(params["out"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        print(f"params['out'] hash_value={hash_value}")

        out_pa_asm = torch.empty_like(params["out"])
        out_pa_asm[:] = 1

        ops.paged_attention_rocm(
            out=params["out"],
            # out=out_pa_asm,
            exp_sum=params["exp_sum"],
            max_logits=params["max_logits"],
            tmp_out=params["tmp_out"],
            query=params["query"],
            key_cache=params["key_cache"],
            value_cache=params["value_cache"],
            num_kv_heads=params["num_kv_heads"],
            scale=params["scale"],
            block_tables=params["block_tables"],
            seq_lens=params["seq_lens"],
            query_start_loc=params["query_start_loc"],
            block_size=params["block_size"],
            max_seq_len=params["max_seq_len"],
            alibi_slopes=params["alibi_slopes"],
            kv_cache_dtype=params["kv_cache_dtype"],
            k_scale=params["k_scale"],
            v_scale=params["v_scale"],
            fp8_out_scale=params["fp8_out_scale"],
        )
        print(params["out"].reshape(-1)[:10])
        hash_value = hashlib.md5(params["out"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        print(f"params['out']   hash_value={hash_value}")

        print(hashlib.md5(params["query"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
        print(hashlib.md5(params2["query"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
        print(hashlib.md5(params["key_cache"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
        print(hashlib.md5(params2["key_cache"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest())

        paged_attn.forward_decode(
            params2["query"],
            params2["key_cache"],
            params2["value_cache"],
            params2["block_tables"],
            params2["seq_lens"],
            params2["max_seq_len"],
            params2["kv_cache_dtype"],
            params2["num_kv_heads"],
            params2["scale"],
            params2["alibi_slopes"],
            params2["k_scale"],
            params2["v_scale"],
            output=out_pa_asm,
        )
        print(params2["out"].reshape(-1)[:10])
        print(out_pa_asm.reshape(-1)[:10])
        hash_value = hashlib.md5(params2["out"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        print(f"params2['out']  hash_value={hash_value}")
        hash_value = hashlib.md5(out_pa_asm.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        print(f"out_pa_asm      hash_value={hash_value}")

        print("out_pa_asm       data_ptr:", out_pa_asm.data_ptr())
        print("params['out']    data_ptr:", params["out"].data_ptr())


load_and_run_test()
