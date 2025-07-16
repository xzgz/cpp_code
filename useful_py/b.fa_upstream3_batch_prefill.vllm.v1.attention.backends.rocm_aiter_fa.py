# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import (
    make_local_attention_virtual_batches)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import os
import sys
import json
import numpy as np
global_count = 0
num_hidden_layers = 32
selected_layer_id = [0, 31]
# root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/flash_attn_varlen_func_impl_dump_data/Llama-3.1-8B-Instruct-FP8-KV/"
root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/flash_attn_varlen_func_impl_dump_data/Llama-3.1-8B-Instruct-FP8-KV/v2_format_prompts_1000_part1/"


def print_tensor_info(name, value):
    print(f"{name}:")
    print(f"  Shape: {tuple(value.shape)}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Device: {value.device}")
    print(f"  Is contiguous: {value.is_contiguous()}")
    print(f"  Stride: {value.stride()}")


def dump_function_inputs(dump_dir, skip_data_tensor_name_list, **kwargs):
    """
    """
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
                    tensor_np = value.detach().cpu().numpy()
                    np.save(file_path, tensor_np)
                else:
                    print(f"tensor {name} only save meta info to txt")

                num_elements = value.numel()
                log.write(f"  Num elements: {num_elements}\n")

                if num_elements <= 10:
                    log.write(f"  Data: {tensor_np.tolist()}\n")
                else:
                    if name not in skip_data_tensor_name_list:
                        flat_data = tensor_np.ravel()
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


if current_platform.is_rocm():
    import aiter

    from vllm.triton_utils import tl, triton
    from vllm.utils import direct_register_custom_op

    def convert_block_table(
        block_table: torch.Tensor,
        block_size: int,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:

        batches, pages = block_table.shape
        kv_page_indices = torch.empty([cu_seqlens_k[-1]], dtype=torch.int32, device="cuda")

        cur_pos = 0
        for b_idx in range(batches):
            for p_idx in range(pages):
                if cur_pos >= cu_seqlens_k[b_idx + 1]:
                    break

                start = cur_pos
                end = min(cur_pos + block_size, cu_seqlens_k[b_idx + 1])

                block_start_pos = block_table[b_idx, p_idx] * block_size
                block_end_pos = block_start_pos + end - start
                kv_page_indices[start:end] = torch.arange(
                    block_start_pos,
                    block_end_pos,
                    dtype=torch.int32,
                    device="cuda",
                )

                cur_pos = end
        return kv_page_indices

    @triton.jit
    def _vllm_layout_trans_kernel(
        k_buffer_ptr,
        v_buffer_ptr,
        k_values_ptr,
        v_values_ptr,
        b_seq_lens_loc,
        block_table,
        block_table_stride_0,
        E_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        batch_token_indexes = tl.load(b_seq_lens_loc + batch_idx +
                                      tl.arange(0, 2))
        batch_token_start, batch_token_end = tl.split(batch_token_indexes)
        seq_len = batch_token_end - batch_token_start
        if block_idx * BLOCK_SIZE < seq_len:
            block_mask = (block_idx * BLOCK_SIZE +
                          tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len

            kv_idx = tl.load(block_table + batch_idx * block_table_stride_0 +
                             block_idx)

            kv_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
                0, BLOCK_SIZE)[:, None] * E_DIM + tl.arange(0, E_DIM)[None, :]
            k_vals = tl.load(k_buffer_ptr + kv_buffer_off,
                             mask=block_mask,
                             other=0.0)
            v_vals = tl.load(v_buffer_ptr + kv_buffer_off,
                             mask=block_mask,
                             other=0.0)

            kv_values_off = batch_token_start * E_DIM + \
                block_idx * BLOCK_SIZE * E_DIM + \
                tl.arange(0, BLOCK_SIZE)[:, None] * E_DIM + \
                tl.arange(0, E_DIM)[None, :]
            tl.store(k_values_ptr + kv_values_off, k_vals, mask=block_mask)
            tl.store(v_values_ptr + kv_values_off, v_vals, mask=block_mask)

    def vllm_layout_trans(b_seq_lens_loc, block_table, k_buffer, v_buffer,
                          max_seq_len, total_tokens):
        H_KV = v_buffer.shape[2]
        D = v_buffer.shape[3]
        BLOCK_SIZE = v_buffer.shape[1]
        dtype = k_buffer.dtype
        k_values = torch.empty((total_tokens, H_KV, D),
                               dtype=dtype,
                               device="cuda")
        v_values = torch.empty((total_tokens, H_KV, D),
                               dtype=dtype,
                               device="cuda")

        grid = (block_table.shape[0],
                (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

        _vllm_layout_trans_kernel[grid](k_buffer,
                                        v_buffer,
                                        k_values,
                                        v_values,
                                        b_seq_lens_loc,
                                        block_table,
                                        block_table.stride(0),
                                        E_DIM=H_KV * D,
                                        BLOCK_SIZE=BLOCK_SIZE)

        return k_values, v_values

    def flash_attn_varlen_func_impl(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        total_tokens: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: Optional[list[int]],  # -1 means infinite context window
        alibi_slopes: Optional[list[float]],
        block_table: torch.Tensor,
        kv_page_indices: torch.Tensor,
    ) -> torch.Tensor:

        # print(f"AAAA=flash_attn_varlen_func_impl")
        # global global_count
        # skip_data_tensor_name_list = ["q", "k", "v", "out"]
        # request_id = global_count // num_hidden_layers
        # layer_id = global_count % num_hidden_layers
        # if layer_id in selected_layer_id:
        #     print(f"global_count={global_count}", flush=True)
        #     print(f"request_id={request_id}", flush=True)
        #     print(f"layer_id={layer_id}", flush=True)

        #     dump_dir = root_data_dir + "prompt_" + str(request_id) + "/layer_" + str(layer_id)
        #     dump_function_inputs(
        #         dump_dir,
        #         skip_data_tensor_name_list,
        #         q=q,
        #         k=k_cache,
        #         v=v_cache,
        #         cu_seqlens_q=cu_seqlens_q,
        #         kv_indptr=cu_seqlens_k,
        #         kv_page_indices=block_table,
        #         kv_page_indices_vllm=kv_page_indices,
        #         total_tokens=total_tokens,
        #         max_seqlen_q=max_seqlen_q,
        #         max_seqlen_k=max_seqlen_k,
        #         softmax_scale=softmax_scale,
        #         causal=True,
        #         is_chunked_prefill=True,
        #         alibi_slopes=alibi_slopes,
        #         window_size=window_size,
        #         out=out,
        #     )

        # global_count += 1


        # k, v = vllm_layout_trans(cu_seqlens_k, block_table, k_cache, v_cache,
        #                          max_seqlen_k, total_tokens)
        # output = aiter.flash_attn_varlen_func(
        #     q=q,
        #     k=k,
        #     v=v,
        #     cu_seqlens_q=cu_seqlens_q,
        #     max_seqlen_q=max_seqlen_q,
        #     cu_seqlens_k=cu_seqlens_k,
        #     max_seqlen_k=max_seqlen_k,
        #     softmax_scale=softmax_scale,
        #     causal=True,
        #     alibi_slopes=alibi_slopes,
        #     window_size=window_size,
        #     out=out,
        # )

        output = aiter.mha_batch_prefill_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=cu_seqlens_k,
            kv_page_indices=block_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            is_chunked_prefill=True,
            alibi_slopes=alibi_slopes,
            window_size=window_size,
            out=out,
        )

        return output

    def flash_attn_varlen_func_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        total_tokens: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: Optional[list[int]],  # -1 means infinite context window
        alibi_slopes: Optional[list[float]],
        block_table: torch.Tensor,
        kv_page_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(q.shape[0],
                           q.shape[1],
                           v_cache.shape[-2],
                           dtype=torch.float8_e4m3fnuz,
                           device="cuda")

    direct_register_custom_op("flash_attn_varlen_func",
                              flash_attn_varlen_func_impl, ["out"],
                              flash_attn_varlen_func_fake)
    flash_attn_varlen_func = torch.ops.vllm.flash_attn_varlen_func

logger = init_logger(__name__)


class AiterFlashAttentionMetadataBuilder:

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        model_config = runner.model_config

        self.runner = runner
        self.num_heads_q = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        total_tokens = self.runner.seq_lens_np[:num_reqs].sum()
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)

        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        cu_seq_lens = torch.zeros(seq_lens.shape[0] + 1,
                                  dtype=torch.int32,
                                  device="cuda")
        torch.cumsum(seq_lens,
                     dim=0,
                     dtype=cu_seq_lens.dtype,
                     out=cu_seq_lens[1:])

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            return None

        # for local attention
        local_attn_metadata = None
        if self.runner.attention_chunk_size is not None:
            seqlens_q_local_np, virt_q_cu_seqlens_np, virt_k_seqlens_np, \
                virt_block_table_tensor = make_local_attention_virtual_batches(
                    self.runner.attention_chunk_size,
                    self.runner.query_start_loc_np[:num_reqs + 1],
                    self.runner.seq_lens_np[:num_reqs],
                    block_table_tensor,
                    self.block_size,
                )
            local_query_start_loc = torch.from_numpy(virt_q_cu_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_seqused_k = torch.from_numpy(virt_k_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_max_query_len = seqlens_q_local_np.max()
            local_max_seq_len = virt_k_seqlens_np.max()
            local_scheduler_metadata = schedule(
                batch_size=local_query_start_loc.shape[0] - 1,
                cu_query_lens=local_query_start_loc,
                max_query_len=local_max_query_len,
                seqlens=local_seqused_k,
                max_seq_len=local_max_seq_len,
                causal=True)

            local_attn_metadata = \
            AiterFlashAttentionMetadata.LocalAttentionMetadata(
                local_query_start_loc=local_query_start_loc,
                local_seqused_k=local_seqused_k,
                local_block_table=virt_block_table_tensor,
                local_max_query_len=local_max_query_len,
                local_max_seq_len=local_max_seq_len,
                local_scheduler_metadata=local_scheduler_metadata,
            )

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.runner.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.runner.device)
            suffix_kv_lens = (self.runner.seq_lens_np[:num_reqs] -
                              common_prefix_len)
            suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(
                self.runner.device)
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=True)

        kv_page_indices = convert_block_table(
            block_table_tensor,
            self.block_size,
            cu_seq_lens,
        )

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            cu_seq_lens=cu_seq_lens,
            total_tokens=total_tokens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            kv_page_indices=kv_page_indices,
            local_attn_metadata=local_attn_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class AiterFlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        return AiterFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AiterFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        return AiterFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AiterFlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    cu_seq_lens: torch.Tensor
    total_tokens: int
    block_table: torch.Tensor
    kv_page_indices: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None

    # for local attention
    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor
        local_seqused_k: torch.Tensor
        local_block_table: torch.Tensor
        local_max_query_len: int
        local_max_seq_len: int
        local_scheduler_metadata: Optional[torch.Tensor]

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


class AiterFlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0.
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = \
            AiterFlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")
        self.use_irope = use_irope
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens] and
        # value[:num_actual_tokens] because the reshape_and_cache_flash op uses
        # the slot_mapping's shape to determine the number of actual tokens.
        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(torch.float8_e4m3fnuz)
            value_cache = value_cache.view(torch.float8_e4m3fnuz)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        # Compute attention and update output up to `num_actual_tokens`.
        use_local_attn = \
            (self.use_irope and attn_metadata.local_attn_metadata is not None)

        if not attn_metadata.use_cascade or use_local_attn:
            if use_local_attn:
                assert attn_metadata.local_attn_metadata is not None
                local_metadata = attn_metadata.local_attn_metadata
                cu_seqlens_q = local_metadata.local_query_start_loc
                seqused_k = local_metadata.local_seqused_k
                max_seqlen_q = local_metadata.local_max_query_len
                max_seqlen_k = local_metadata.local_max_seq_len
                block_table = local_metadata.local_block_table
            else:
                cu_seqlens_q = attn_metadata.query_start_loc
                seqused_k = attn_metadata.seq_lens
                max_seqlen_q = attn_metadata.max_query_len
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table

            if max_seqlen_q > 1:
                cu_seq_lens = attn_metadata.cu_seq_lens
                total_tokens = attn_metadata.total_tokens
                flash_attn_varlen_func(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    total_tokens=total_tokens,
                    softmax_scale=self.scale,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=block_table,
                    cu_seqlens_k=cu_seq_lens,
                    kv_page_indices=attn_metadata.kv_page_indices,
                )

            _, num_heads, head_size = query.shape
            _PARTITION_SIZE_ROCM = 256
            num_seqs = seqused_k.shape[0]
            nbyes_per_qo_elem = torch.finfo(output.dtype).bits // 8
            max_num_partitions = (max_seqlen_k + _PARTITION_SIZE_ROCM -
                                  1) // _PARTITION_SIZE_ROCM

            workspace_buffer = torch.empty(
                (num_seqs * num_heads * max_num_partitions * head_size) *
                nbyes_per_qo_elem + 2 *
                (num_seqs * num_heads * max_num_partitions) * 4,
                dtype=torch.uint8,
                device=output.device,
            )

            aiter.paged_attention_v1(
                output[:num_actual_tokens],
                workspace_buffer,
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self.scale,
                block_table,
                cu_seqlens_q,
                seqused_k,
                int(max_seqlen_k),
                self.alibi_slopes,
                self.kv_cache_dtype,
                "NHD",
                self.logits_soft_cap,
                layer._k_scale,
                layer._v_scale,
                None,
                _PARTITION_SIZE_ROCM,
            )
            return output
        else:
            raise NotImplementedError(
                "Cascade attention is not implemented for ROCM AITER")
