#include "engine/npu_fused_target.h"
#include <cstdio>
int main() {
    const char* kModel = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char* kXclbin = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-persistent-token127";
    const char* kWeights = "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref";
    int32_t ids[] = {1,6,12,18,24};
    NpuFusedTarget target(kModel, kXclbin, kWeights, ids, 5);
    int32_t token = 151643;
    std::vector<float> logits(151936), hidden(28672);
    target.forward(&token, 1, logits.data(), hidden.data());
    float mn=1e9,mx=-1e9; int nz=0;
    for(int i=0;i<28672;i++){if(hidden[i]<mn)mn=hidden[i];if(hidden[i]>mx)mx=hidden[i];if(hidden[i]!=0)nz++;}
    bool nan=false; for(int i=0;i<28672;i++)if(std::isnan(hidden[i])){nan=true;break;}
    printf("Hidden: min=%.2f max=%.2f nonzero=%d/28672 NaN=%s\n",mn,mx,nz,nan?"YES":"no");
    return nan?1:0;
}
