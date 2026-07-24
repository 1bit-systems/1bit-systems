__global__ void fixup_skip_expert_kernel(int* __restrict__ expert_idx, int* __restrict__ skip_flag, int n_exp, int n_exp_plus1){
    int idx=*expert_idx;
    if(idx==n_exp||idx>=n_exp_plus1){*expert_idx=0; *skip_flag=1;}
    else{*skip_flag=0;}
}
