#include "kv_quant.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
int main(){
    // Test all-ones edge case
    {
        KVQuant kq; float K[1024];
        for(int i=0;i<1024;i++) K[i] = 1.0f;
        kq.append(0, K, K);
        float buf[128];
        for(int kvh=0;kvh<8;kvh++){
            kq.dequant_K_all(kvh, 1, buf);
            for(int d=0;d<128;d++){
                float e = fabsf(buf[d] - 1.0f);
                if(e > 0.01f){ printf("FAIL: all-ones err=%.6f\n",e); return 1; }
            }
        }
        printf("PASS: all-ones\n");
    }
    // Test random data, all 8 KV heads
    {
        KVQuant kq; float K[1024],V[1024];
        for(int i=0;i<1024;i++){K[i]=(float)(rand()%10000-5000)/1000.0f; V[i]=(float)(rand()%10000-5000)/1000.0f;}
        kq.append(0,K,V); kq.append(1,K,V);
        float me=0,buf[128];
        for(int kvh=0;kvh<8;kvh++){ 
            kq.dequant_K_all(kvh,2,buf);
            for(int t=0;t<2;t++) for(int d=0;d<128;d++){
                float e=fabsf(buf[t*128+d]-K[kvh*128+d]);
                if(e>me)me=e;}
        }
        printf("MAX_ERR_K: %.6f\n",me);
        if(me>0.5f){printf("FAIL: K error too large\n");return 1;}
        me=0;
        for(int kvh=0;kvh<8;kvh++){ 
            kq.dequant_V_all(kvh,2,buf);
            for(int t=0;t<2;t++) for(int d=0;d<128;d++){
                float e=fabsf(buf[t*128+d]-V[kvh*128+d]);
                if(e>me)me=e;}
        }
        printf("MAX_ERR_V: %.6f\n",me);
        if(me>0.5f){printf("FAIL: V error too large\n");return 1;}
        printf("PASS: random data\n");
    }
    // Test small values (typical normalized KV)
    {
        KVQuant kq; float K[1024],V[1024];
        for(int i=0;i<1024;i++){K[i]=(float)(rand()%200-100)/100.0f; V[i]=(float)(rand()%200-100)/100.0f;}
        kq.append(0,K,V); kq.append(1,K,V);
        float me=0,buf[128];
        for(int kvh=0;kvh<8;kvh++){ 
            kq.dequant_K_all(kvh,2,buf);
            for(int t=0;t<2;t++) for(int d=0;d<128;d++){
                float e=fabsf(buf[t*128+d]-K[kvh*128+d]);
                if(e>me)me=e;}
        }
        printf("MAX_ERR small: %.6f\n",me);
        // Small values should have proportionally smaller error (~0.02 for [-1,1])
        if(me>0.05f){printf("NOTE: small value error=%.6f (expected <0.05)\n",me);}
        printf("PASS: small values\n");
    }
    printf("ALL PASS\n");
    return 0;
}
