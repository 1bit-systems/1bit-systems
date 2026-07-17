__global__ void gqa_broadcast_k(const __half* __restrict__ v, __half* __restrict__ out, int nq, int nkv, int hd){
    int idx=threadIdx.x+blockIdx.x*blockDim.x; int total=nq*hd; if(idx>=total)return;
    int h=idx/hd,d=idx%hd; int kv=h/(nq/nkv); out[idx]=v[kv*hd+d];
}
