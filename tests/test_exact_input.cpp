#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#define HIP_OK(e) do{auto _s=(e);if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d\n",_s);abort();}}while(0)
constexpr int H=2048,QD=1024,KD=256,QKV=1280,NKV=2;

static const std::string& weights_dir() {
    static std::string dir = [] {
        const char* d = getenv("ZAYA_WEIGHTS_DIR");
        if (d && d[0]) return std::string(d);
        const char* home = getenv("HOME");
        return (home && home[0]) ? std::string(home) + "/.local/share/1bit-systems/weights/" : "/tmp/zaya_weights/";
    }();
    return dir;
}
static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){return {};}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
static std::vector<float> load_weight(const std::string& name){
    return load_bin(weights_dir() + name);
}
static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    HIP_OK(hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h));
}

__global__ void cca_attn_kernel(
    const __half*hs,const __half*phs,const __half*csi,int pos,
    const __half*wq,const __half*wk,const __half*wv1,const __half*wv2,const __half*wo,
    const float*cdw,const float*cdb,const float*cgw,const float*cgb,
    const float*ks,const __half*nw,
    __half*ao,__half*ncs,__half*nph);

int main() {
    printf("Test CCA kernel with EXACT hs from npz\n");
    
    // Load exact hs and nw from npz
    auto hs = load_bin("/tmp/hs_from_npz.bin");
    auto ref = load_bin("/tmp/cca_ref_extracted/scaled.bin");
    if(hs.empty() || ref.empty()){printf("Can't load ref\n");return 1;}
    printf("hs[0]=%.4f ref[0]=%.4f\n",hs[0],ref[0]);
    
    // Load nw from npz
    auto nw = load_bin("/tmp/cca_ref_extracted/nw.bin");
    
    __half *d_hs,*d_nw,*d_out,*d_ncs,*d_nph,*d_wq,*d_wk,*d_wv1,*d_wv2,*d_wo,*d_phs,*d_csi;
    float *d_cdw,*d_cdb,*d_cgw,*d_cgb,*d_ks;
    HIP_OK(hipMalloc(&d_hs,H*2));HIP_OK(hipMalloc(&d_nw,H*2));
    HIP_OK(hipMalloc(&d_out,H*2));HIP_OK(hipMalloc(&d_ncs,QKV*2*2));HIP_OK(hipMalloc(&d_nph,H*2));
    HIP_OK(hipMalloc(&d_wq,QD*H*2));HIP_OK(hipMalloc(&d_wk,KD*H*2));
    HIP_OK(hipMalloc(&d_wv1,(KD/2)*H*2));HIP_OK(hipMalloc(&d_wv2,(KD/2)*H*2));HIP_OK(hipMalloc(&d_wo,H*QD*2));
    HIP_OK(hipMalloc(&d_cdw,QKV*2*4));HIP_OK(hipMalloc(&d_cdb,QKV*4));
    HIP_OK(hipMalloc(&d_cgw,QKV*128*2*4));HIP_OK(hipMalloc(&d_cgb,QKV*4));
    HIP_OK(hipMalloc(&d_ks,NKV*4));HIP_OK(hipMalloc(&d_phs,H*2));HIP_OK(hipMalloc(&d_csi,QKV*2*2));
    
    auto wq=load_weight("model_layers_0_self_attn_qkv_proj_q_proj_weight.bin");
    auto wk=load_weight("model_layers_0_self_attn_qkv_proj_k_proj_weight.bin");
    auto cdw=load_weight("model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_weight.bin");
    auto cdb=load_weight("model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_bias.bin");
    auto cgw=load_weight("model_layers_0_self_attn_qkv_proj_conv_qk_grouped_weight.bin");
    auto cgb=load_weight("model_layers_0_self_attn_qkv_proj_conv_qk_grouped_bias.bin");
    auto ks=load_weight("model_layers_0_self_attn_qk_norm_temp.bin");
    auto wv1=load_weight("model_layers_0_self_attn_qkv_proj_v_proj_current_weight.bin");
    auto wv2=load_weight("model_layers_0_self_attn_qkv_proj_v_proj_delayed_weight.bin");
    auto wo=load_weight("model_layers_0_self_attn_o_proj_weight.bin");

    hipStream_t st;HIP_OK(hipStreamCreate(&st));
    upf16(hs,d_hs,H,st); upf16(nw,d_nw,H,st);
    upf16(wq,d_wq,QD*H,st); upf16(wk,d_wk,KD*H,st);
    upf16(wv1,d_wv1,(KD/2)*H,st); upf16(wv2,d_wv2,(KD/2)*H,st); upf16(wo,d_wo,H*QD,st);
    hipMemcpyAsync(d_cdw,cdw.data(),QKV*2*4,hipMemcpyHostToDevice,st);
    hipMemcpyAsync(d_cdb,cdb.data(),QKV*4,hipMemcpyHostToDevice,st);
    hipMemcpyAsync(d_cgw,cgw.data(),QKV*128*2*4,hipMemcpyHostToDevice,st);
    hipMemcpyAsync(d_cgb,cgb.data(),QKV*4,hipMemcpyHostToDevice,st);
    hipMemcpyAsync(d_ks,ks.data(),NKV*4,hipMemcpyHostToDevice,st);
    hipMemsetAsync(d_phs,0,H*2,st);
    hipMemsetAsync(d_csi,0,QKV*2*2,st);
    hipStreamSynchronize(st);
    
    cca_attn_kernel<<<1,256,0,st>>>(d_hs,d_phs,d_csi,0,d_wq,d_wk,d_wv1,d_wv2,d_wo,d_cdw,d_cdb,d_cgw,d_cgb,d_ks,d_nw,d_out,d_ncs,d_nph);
    hipStreamSynchronize(st);
    
    std::vector<__half> ho(H);
    hipMemcpy(ho.data(),d_out,H*2,hipMemcpyDeviceToHost);
    
    float cos=0,na=0,nb=0;
    for(int i=0;i<H;i++){float gv=__half2float(ho[i]);cos+=gv*ref[i];na+=gv*gv;nb+=ref[i]*ref[i];}
    cos/=(sqrtf(na)*sqrtf(nb)+1e-12f);
    printf("cos_sim = %.6f %s\n",cos,cos>0.99f?"PASS":"FAIL");
    printf("GPU[0:4]: %.4f %.4f %.4f %.4f\n",__half2float(ho[0]),__half2float(ho[1]),__half2float(ho[2]),__half2float(ho[3]));
    printf("Ref[0:4]: %.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3]);
    
    hipFree(d_hs);hipFree(d_nw);hipFree(d_out);hipFree(d_ncs);hipFree(d_nph);
    hipFree(d_wq);hipFree(d_wk);hipFree(d_wv1);hipFree(d_wv2);hipFree(d_wo);
    hipFree(d_cdw);hipFree(d_cdb);hipFree(d_cgw);hipFree(d_cgb);hipFree(d_ks);
    hipFree(d_phs);hipFree(d_csi);hipStreamDestroy(st);
    return cos>0.99f?0:1;
}
