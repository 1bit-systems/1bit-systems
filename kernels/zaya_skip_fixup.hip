// fixup_skip_expert_kernel — GPU-side skip handling for MoE routing.
// When router selects the "skip" expert (idx == n_exp), this kernel:
// 1. Maps expert_idx to 0 (valid expert for gateup/down to use)
// 2. Sets skip_flag = 1 so gateup/down kernels know to output zeros
// 3. Does NOT zero any buffers — gateup/down handle the skip
__global__ void fixup_skip_expert_kernel(int* __restrict__ expert_idx, int* __restrict__ skip_flag, int n_exp, int n_exp_plus1){
    int idx=*expert_idx;
    if(idx==n_exp||idx>=n_exp_plus1){*expert_idx=0; *skip_flag=1;}
    else{*skip_flag=0;}
}