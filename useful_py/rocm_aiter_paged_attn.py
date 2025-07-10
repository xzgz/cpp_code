# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import aiter as rocm_aiter
import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.platforms import current_platform
from vllm.utils import cdiv

FP8_DTYPE = current_platform.fp8_dtype()

import sys
from typing import Any

num_hidden_layers = 80
global_count = 0
prefill_step_id = 0
decoder_step_id = 0
prefill_skip_step_count = 1
decoder_skip_step_count = 50
is_prefill = False

sample_id = 0


def filter_print(msg):
    global global_count
    global prefill_step_id
    global decoder_step_id
    global is_prefill

    layer_id = global_count % num_hidden_layers
    if is_prefill:
        if layer_id == 0:
            if prefill_step_id % prefill_skip_step_count == 0:
                print(msg)
                sys.stdout.flush()
    else:
        if layer_id == 0:
            if decoder_step_id % decoder_skip_step_count == 0:
                print(msg)
                sys.stdout.flush()


def update_global_flag():
    global global_count
    global prefill_step_id
    global decoder_step_id
    global is_prefill
    global sample_id

    layer_id = global_count % num_hidden_layers
    if is_prefill:
        if layer_id == 0:
            decoder_step_id = 0
            prefill_step_id += 1
    else:
        if layer_id == 0:
            if decoder_step_id % decoder_skip_step_count == 0:
                sample_id += 1
            prefill_step_id = 0
            decoder_step_id += 1
    global_count += 1


def dump_function_inputs(dump_dir, skip_data_tensor_name_list, **kwargs):
    import os
    import sys
    import json
    import numpy as np

    """
    """
    size2type_map = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
    }

    os.makedirs(dump_dir, exist_ok=True)
    metadata = {
        'tensors': {},
        'non_tensors': {}
    }

    log_file = os.path.join(dump_dir, 'params_info.txt')
    with open(log_file, 'w') as log:
        for name, value in kwargs.items():
            log.write(f"Parameter: {name}\n")

            if isinstance(value, torch.Tensor):
                log.write(f"  Type: torch.Tensor\n")
                log.write(f"  StorageDataPtr: {value.storage().data_ptr()}\n")
                log.write(f"  DataPtr       : {value.data_ptr()}\n")
                log.write(f"  Shape: {tuple(value.shape)}\n")
                log.write(f"  Dtype: {value.dtype}\n")
                log.write(f"  Device: {value.device}\n")
                log.write(f"  Is contiguous: {value.is_contiguous()}\n")
                log.write(f"  Stride: {value.stride()}\n")

                tensor_np = None
                if name not in skip_data_tensor_name_list:
                    print(f"tensor {name} save meta info to txt and data to disk")
                    file_path = os.path.join(dump_dir, f"{name}.npy")
                    write_data_type = size2type_map[value.element_size()]
                    tensor_np = value.view(write_data_type).detach().cpu().numpy()
                    np.save(file_path, tensor_np)
                else:
                    print(f"tensor {name} only save meta info to txt")

                num_elements = value.numel()
                log.write(f"  Num elements: {num_elements}\n")

                if num_elements <= 10:
                    log.write(f"  Data: {tensor_np.tolist()}\n")
                else:
                    if name not in skip_data_tensor_name_list:
                        flat_data = value.to(torch.float32).ravel().detach().cpu().numpy()
                        head = flat_data[:5].tolist()
                        tail = flat_data[-5:].tolist()
                        log.write(f"  Data (partial): head={head}, tail={tail}\n")
                        log.write(f"  Data range: min={np.min(tensor_np):.4f}, max={np.max(tensor_np):.4f}, mean={np.mean(tensor_np):.4f}\n")

                metadata['tensors'][name] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'file': f"{name}.npy",
                    'stride': value.stride(),
                    'is_contiguous': value.is_contiguous()
                }
            else:
                log.write(f"  Type: {type(value).__name__}\n")

                if value is None:
                    log.write(f"  Value: None\n")
                    metadata['non_tensors'][name] = None
                elif isinstance(value, tuple) or isinstance(value, list):
                    log.write(f"  Value[0] type: {type(value[0]).__name__}\n")
                    log.write(f"  Value: {value}\n")
                    metadata['non_tensors'][name] = list(value)
                else:
                    log.write(f"  Value: {value}\n")
                    metadata['non_tensors'][name] = value
        log.flush()

    with open(os.path.join(dump_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Parameter exported to: {dump_dir}")
    print(f"Detailed information refer to: {log_file}")
    sys.stdout.flush()


def filter_dump(root_data_dir, skip_data_tensor_name_list, **kwargs):
    global global_count
    global decoder_step_id
    global sample_id

    layer_id = global_count % num_hidden_layers
    if layer_id == 0:
        if decoder_step_id % decoder_skip_step_count == 0:
            if sample_id < 25:
                dump_dir = root_data_dir + "/" + str(sample_id) + "/"
                dump_function_inputs(dump_dir, skip_data_tensor_name_list, **kwargs)


class AITERPagedAttention(PagedAttention):
    is_asm_supported: bool = False

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        if not AITERPagedAttention.is_asm_supported:
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            kv_cache_torch_dtype = FP8_DTYPE \
                        if "fp8" in kv_cache_dtype else torch.int8
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

            # rocm_aiter.reshape_and_cache_with_pertoken_quant(
            #     key, value, key_cache, value_cache, k_scale, v_scale,
            #     slot_mapping.flatten(), True)
            rocm_aiter.reshape_and_cache(key, value, key_cache, value_cache,
                                         slot_mapping.flatten(),
                                         kv_cache_dtype, k_scale, v_scale,
                                         True)

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        if not AITERPagedAttention.is_asm_supported:
            filter_print(f"AAAA=not AITERPagedAttention.is_asm_supported")
            import aiter

            max_num_partitions = (max_seq_len + 256 - 1) // 256
            assert 256 % block_size == 0
            num_seqs, num_heads, head_size = query.shape
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            return aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                None,
                256,
            )

        if "fp8" in kv_cache_dtype:
            kv_cache_torch_dtype = FP8_DTYPE
            # kv_cache_torch_dtype = torch.int8
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0
                    and blocksparse_block_size % block_size == 0), (
                        f"{blocksparse_block_size=} needs to be a multiple of"
                        f"{block_size=} used in block_tables.")

        max_num_blocks_per_seq = cdiv(max_seq_len, block_size)

        filter_print(f"AAAA=rocm_aiter.pa_fwd_asm")
        filter_print(f"key_cache.dtype={key_cache.dtype}")
        filter_print(f"block_tables.dtype={block_tables.dtype}")
        filter_print(f"query.shape={query.shape}")
        filter_print(f"key_cache.shape={key_cache.shape}")
        filter_print(f"value_cache.shape={value_cache.shape}")
        filter_print(f"block_tables.shape={block_tables.shape}")
        filter_print(f"seq_lens.shape={seq_lens.shape}")
        filter_print(f"seq_lens={seq_lens}")
        filter_print(f"max_num_blocks_per_seq={max_num_blocks_per_seq}")

        # root_data_dir = "/lab-mlperf-inference/code/op_pa_input/llama2_70b/dk2_eager_step1"
        root_data_dir = "/lab-mlperf-inference/code/op_pa_input/llama2_70b/dk2_eager_step11"
        skip_data_tensor_name_list = ["query", "key_cache", "value_cache"]
        # filter_dump(
        #     root_data_dir,
        #     skip_data_tensor_name_list,
        #     query=query,
        #     key_cache=key_cache,
        #     value_cache=value_cache,
        #     block_tables=block_tables,
        #     context_lens=seq_lens,
        #     max_num_blocks=max_num_blocks_per_seq,
        #     K_QScale=k_scale,
        #     V_QScale=v_scale,
        #     out_=output,
        # )

        update_global_flag()

        rocm_aiter.pa_fwd_asm(
            query,
            key_cache,
            value_cache,
            # asm_V_shuffle(value_cache),
            block_tables,
            seq_lens,
            max_num_blocks_per_seq,
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=output,
        )
        return output
