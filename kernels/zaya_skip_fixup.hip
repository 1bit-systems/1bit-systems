__global__ void fixup_skip_expert_kernel(int* __restrict__ expert_idx, __half* __restrict__ d_tmp, int* __restrict__ skip_flag, int n_exp, int n_exp_plus1, int h){
    int idx=*expert_idx;
    if(idx==n_exp||idx>=n_exp_plus1){*expert_idx=0; for(int i=threadIdx.x;i<h;i+=blockDim.x)d_tmp[i]=__float2half(0.0f); *skip_flag=1;}
    else{*skip_flag=0;}
}
