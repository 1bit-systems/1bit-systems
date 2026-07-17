#include "kv_quant.h"
KVQuant::KVQuant()
 : k_data(KQ_MAX_CTX*KQ_ELEM/2,0), v_data(KQ_MAX_CTX*KQ_ELEM/2,0),
   k_scales(KQ_MAX_CTX*KQ_GROUPS,{0,0}), v_scales(KQ_MAX_CTX*KQ_GROUPS,{0,0}), n(0) {}
void KVQuant::quantize_group(const float* s, uint8_t* p, KQScale* gs) {
    float mn=1e30f,mx=-1e30f; for(int i=0;i<KQ_GROUP;i++){if(s[i]<mn)mn=s[i];if(s[i]>mx)mx=s[i];}
    // Handle constant values (range=0)
    if(mx <= mn) {
        // All values are equal. Quantize to q0=0, scale=1, zp=val.
        float sc = 1.0f; float zp = s[0];
        gs->scale=f32bf16(sc); gs->zp=f32bf16(zp);
        // Store q=0 for all elements
        for(int i=0;i<KQ_GROUP;i+=2) p[i/2]=0;
        return;
    }
    float sc=(mx-mn)/15.0f; float zp=mn+8.0f*sc;
    gs->scale=f32bf16(sc); gs->zp=f32bf16(zp); float inv=1.0f/sc;
    for(int i=0;i<KQ_GROUP;i+=2){
        int q0=(int)roundf((s[i]-mn)/sc-8.0f),q1=i+1<KQ_GROUP?(int)roundf((s[i+1]-mn)/sc-8.0f):0;
        q0=std::max(-8,std::min(7,q0)); q1=std::max(-8,std::min(7,q1));
        p[i/2]=(uint8_t)((q1&0xF)<<4)|(uint8_t)(q0&0xF);}
}
void KVQuant::dequant_group(const uint8_t* p, const KQScale& gs, float* d) {
    float sc=bf16f32(gs.scale), zp=bf16f32(gs.zp);
    for(int i=0;i<KQ_GROUP;i+=2){
        uint8_t b=p[i/2]; int q0=(int8_t)(b&0x0F); if(q0>=8)q0-=16;
        int q1=(int8_t)(b>>4); if(q1>=8)q1-=16;
        d[i]=(float)q0*sc+zp; if(i+1<KQ_GROUP) d[i+1]=(float)q1*sc+zp;}
}
void KVQuant::append(int pos, const float*K, const float*V) {
    for(int g=0;g<KQ_GROUPS;g++){
        int bo=pos*KQ_ELEM/2+g*KQ_GROUP/2, so=pos*KQ_GROUPS+g;
        quantize_group(K+g*KQ_GROUP,&k_data[bo],&k_scales[so]);
        quantize_group(V+g*KQ_GROUP,&v_data[bo],&v_scales[so]);}
    if(pos>=n)n=pos+1;
}
void KVQuant::dequant_K_all(int kvh, int nt, float* out) const {
    int gpk=KQ_HD/KQ_GROUP;
    for(int t=0;t<nt;t++) for(int gi=0;gi<gpk;gi++){
        int gi_=t*KQ_GROUPS+kvh*gpk+gi;
        int bo=t*KQ_ELEM/2+(kvh*KQ_HD+gi*KQ_GROUP)/2;
        dequant_group(&k_data[bo],k_scales[gi_],out+t*KQ_HD+gi*KQ_GROUP);}
}
void KVQuant::dequant_V_all(int kvh, int nt, float* out) const {
    int gpk=KQ_HD/KQ_GROUP;
    for(int t=0;t<nt;t++) for(int gi=0;gi<gpk;gi++){
        int gi_=t*KQ_GROUPS+kvh*gpk+gi;
        int bo=t*KQ_ELEM/2+(kvh*KQ_HD+gi*KQ_GROUP)/2;
        dequant_group(&v_data[bo],v_scales[gi_],out+t*KQ_HD+gi*KQ_GROUP);}
}
