--- a/engine/npu/src/npu_engine_universal.cpp
+++ b/engine/npu/src/npu_engine_universal.cpp
@@ -65,7 +65,10 @@
     bool init(xrt::device&d,const char*xp,const char*ip,int gid_B,int nlayers){
         NL=nlayers;FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
-        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
+        ins.resize(sz/4);
+        if (fread(ins.data(), 4, ins.size(), f) != ins.size()) {
+            fprintf(stderr, "truncated instruction file\n"); fclose(f); return false;
+        }
+        fclose(f);
         xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
         hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
         bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
@@ -107,7 +110,12 @@
     // Wait for run, sync C back, and dequantize.
     inline void dequantize(xrt::run& r,float*C,int am,int an,float ascale,float Bscale){
-        r.wait();
+        try { r.wait(); }
+        catch (const std::exception& e) {
+            fprintf(stderr, "NPU error (dequantize): %s\n", e.what());
+            memset(C, 0, (size_t)am * an * sizeof(float));
+            return;
+        }
         bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
         float cs=ascale*Bscale;
         for(int m=0;m<am;m++)for(int n=0;n<an;n++){
@@ -117,7 +125,12 @@
     // Wait for NPU kernel completion without readback.
     // Returns immediately after kernel finishes. Call sync_back_and_dequant() later.
     inline void wait_kernel(xrt::run& r){
-        r.wait();
+        try { r.wait(); }
+        catch (const std::exception& e) {
+            fprintf(stderr, "NPU error (wait_kernel): %s\n", e.what());
+            // Caller must handle the error; we can't return an error code here.
+            // Zero the output in dequantize if wait fails.
+        }
     }
     // Sync C back from device and dequantize (call after wait_kernel).
     // This is CPU-only work that CAN overlap with the next kernel's NPU execution.
@@ -158,7 +171,11 @@
     // Complete an async launch: wait + dequant
     inline void finish_async(xrt::run& r,float*C,int am,int an,float ascale,float Bscale){
-        r.wait();
+        try { r.wait(); }
+        catch (const std::exception& e) {
+            fprintf(stderr, "NPU error (finish_async): %s\n", e.what());
+            return;
+        }
         dequantize(r,C,am,an,ascale,Bscale);
     }
 };
@@ -204,7 +221,7 @@
     const char*mp=argv[1];int ng=(argc>2&&!worker_mode)?atoi(argv[2]):32;if(ng<1)ng=1;
-    if(ng>4096)ng=4096; // cap to KV cache size (issue #112)
+    if(ng>16384)ng=16384; // cap to KV cache size (issue #112)
     const char*input_tok_file=(argc>3&&!worker_mode&&argv[3][0]!='\0')?argv[3]:nullptr;
 
     // Model tag
@@ -327,7 +344,7 @@
     // v12: M=32 batch decode
     int BS=32;
     struct KVCache{std::vector<float>k,v;int n;KVCache(int size):k(size),v(size),n(0){}};
-    int kv_size=4096*NKV*HD;
+    int kv_size=16384*NKV*HD;
     std::vector<KVCache> kv_caches;for(int i=0;i<NC;i++)kv_caches.emplace_back(kv_size);
 
     // Workers
@@ -425,7 +442,11 @@
                 double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                 for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+pi);
-                memcpy(&kv_caches[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_caches[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}}
+                if ((size_t)(sp+pi+1)*NKV*HD <= (size_t)kv_caches[l].k.size()) {
+                    memcpy(&kv_caches[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);
+                    memcpy(&kv_caches[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);
+                } else {
+                    fprintf(stderr, "KV cache overflow at layer %d pos %d\n", l, sp+pi);
+                }}}
         kv_caches[l].n=sp+npt;int cl=kv_caches[l].n;
         // Causal attention: token pi attends only to positions [0, sp+pi]
         for(int pi=0;pi<npt;pi++){fprintf(stderr,"a");fflush(stderr);attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA,sp+pi+1);}