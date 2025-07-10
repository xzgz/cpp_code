export HIP_VISIBLE_DEVICES=6

# export VLLM_ROCM_USE_AITER=1
# export VLLM_USE_V1=1
export PYTHONPATH=/data/heyanguang/code/vllm_fa_batch_prefill/rocm_vllm:$PYTHONPATH


python3 -u /mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/useful_py/test_vllm_model.py
