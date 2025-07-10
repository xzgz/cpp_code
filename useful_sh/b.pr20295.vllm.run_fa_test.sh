export PYTHONPATH=/data/heyanguang/code/vllm_fa_batch_prefill/vllm:$PYTHONPATH

pytest ./tests/kernels/attention/test_aiter_flash_attn.py::test_varlen_with_paged_kv -v -s -k ""

# VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 
python3 -u ./tests/kernels/attention/test_aiter_flash_attn.py
