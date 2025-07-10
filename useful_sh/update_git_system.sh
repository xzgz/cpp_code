set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'

alias gst='git status'
alias gsth='git status | head'
alias gstt='git status | tail'
alias glog='git log | head'
alias gb='git branch'
alias gc='git checkout'
alias fomm='git fetch origin master:master'
alias rebhm='git rebase HEAD^ HEAD --onto=master'
alias phhd='git push origin HEAD:refs/for/master'

RED_BOLD='\033[1;31m'
BLUE_BOLD='\e[1;34m'
NC='\033[0m'

export CODE_ROOT_DIR=/mnt/raid0/heyanguang/code


function show_git_repository_status {
    pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/vllm
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    gst
    gb
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/rocm_vllm
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    gst
    gb
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/aiter
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    gst
    gb
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    pushd $CODE_ROOT_DIR/aiter
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    gst
    gb
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    pushd $CODE_ROOT_DIR/cpp_code
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    gst
    gb
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

}

function update_git_repository_status {
    # pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/vllm
    # echo -e "${BLUE_BOLD}$PWD START${NC}"
    # cp ./vllm/attention/backends/rocm_flash_attn.py ./vllm/attention/ops/rocm_aiter_paged_attn.py ../useful_py/
    # echo -e "${BLUE_BOLD}$PWD END${NC}"
    # popd

    # pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/rocm_vllm
    # echo -e "${BLUE_BOLD}$PWD START${NC}"
    # git commit --am --no-edit
    # # git push origin HEAD:fa_upstream3_batch_prefill -f
    # git push my_repo HEAD:rocm_vllm_fa_upstream3_batch_prefill -f
    # echo -e "${BLUE_BOLD}$PWD END${NC}"
    # popd

    # pushd $CODE_ROOT_DIR/vllm_fa_batch_prefill/aiter
    # echo -e "${BLUE_BOLD}$PWD START${NC}"
    # cp ./op_tests/test_batch_prefill.py ../useful_py/
    # cp ./run.sh ../useful_sh/b.vllm_v1_pa_batch_prefill.aiter.run.sh

    # git add ./run.sh ./op_tests/test_batch_prefill.py
    # git commit --am --no-edit
    # # git push origin HEAD:vllm_v1_pa_batch_prefill -f
    # git push my_repo HEAD:aiter_vllm_v1_pa_batch_prefill -f
    # echo -e "${BLUE_BOLD}$PWD END${NC}"
    # popd

    pushd $CODE_ROOT_DIR
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    ls -lah ./vllm_fa_batch_prefill/useful_py/*.npy
    sudo rm -f ./vllm_fa_batch_prefill/useful_py/*.npy
    cp -r ./vllm_fa_batch_prefill/useful_* ./cpp_code/
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    pushd $CODE_ROOT_DIR/cpp_code
    echo -e "${BLUE_BOLD}$PWD START${NC}"
    git add .
    git commit --am --no-edit
    git push origin HEAD:main -f
    echo -e "${BLUE_BOLD}$PWD END${NC}"
    popd

    # pushd $CODE_ROOT_DIR/aiter
    # echo -e "${BLUE_BOLD}$PWD START${NC}"
    # git fetch origin main:main
    # # git add ./log_run/att.txt
    # # git commit --am --no-edit
    # # # git push origin HEAD:skinny_gemm_support_bf16_tmp -f
    # # git push my_repo HEAD:aiter_skinny_gemm_support_bf16_tmp -f
    # # # git push my_repo HEAD:aiter_opt_skinny_gemm_quant -f
    # echo -e "${BLUE_BOLD}$PWD END${NC}"
    # popd

    # pushd $CODE_ROOT_DIR/poc_kl
    # echo -e "${BLUE_BOLD}$PWD START${NC}"
    # cp /mnt/raid0/heyanguang/code/ds_out/PA_A16W8_Q8_1TG_4W_16mx1_64nx4_MTP.ds_explain.md ./doc
    # cp /mnt/raid0/heyanguang/code/ds_out/MLA_A16W16_1TG_4W_16mx4_16nx1_Coex0_Msk1_QH16.ds_explain.md ./doc
    # git add ./doc
    # # git commit -m "add DS explain of MLA and PA asm code"
    # git commit --am --no-edit
    # git push my_repo HEAD:main -f
    # echo -e "${BLUE_BOLD}$PWD END${NC}"
    # popd

}

# show_git_repository_status
update_git_repository_status

set +x
