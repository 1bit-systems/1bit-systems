// gguf_to_h1b — Convert Bonsai F16 GGUF → .h1b + sidecar .gguf
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
};

class GgufReader {
public:
    std::ifstream f;
    uint64_t data_start = 0;
    uint32_t version = 0, alignment = 32;
    std::string arch;
    uint32_t hs=0, is_=0, layers=0, heads=0, kv=0, vocab=0, maxseq=0;
    float rope_theta = 1000000.0f;
    std::map<std::string,GgufTensor> tensors;

    bool open(const std::string& path) {
        f.open(path, std::ios::binary); if (!f) return false;
        char magic[4]; f.read(magic,4);
        if (std::strncmp(magic,"GGUF",4)!=0) return false;
        auto ru32=[&](uint32_t&v){f.read((char*)&v,4);return!!f;};
        auto ru64=[&](uint64_t&v){f.read((char*)&v,8);return!!f;};
        auto rstr=[&](std::string&s){
            uint64_t l; if(!ru64(l))return false;
            if(l>0){s.resize((size_t)l);f.read(&s[0],l);}else s.clear();
            return!!f;
        };
        auto skip=[&](uint32_t vt){
            switch(vt){
                case 0:case 1:case 7:f.seekg(1,std::ios::cur);break;
                case 2:case 3:f.seekg(2,std::ios::cur);break;
                case 4:case 5:case 6:f.seekg(4,std::ios::cur);break;
                case 8:{std::string s;rstr(s);}break;
                case 9:{uint32_t at;ru32(at);uint64_t al;ru64(al);
                    if(at==8)for(uint64_t j=0;j<al;++j){std::string s;rstr(s);}
                    else f.seekg((long)(al*4),std::ios::cur);
                }break;
                case 10:case 11:case 12:f.seekg(8,std::ios::cur);break;
            }
        };

        ru32(version);
        uint64_t nt, nk; ru64(nt); ru64(nk);
        for(uint64_t i=0;i<nk;++i){
            std::string k; rstr(k); uint32_t vt; ru32(vt);
            if(vt==8){std::string v;rstr(v);
                if(k=="general.architecture")arch=v;
                else if(k=="qwen3.hidden_size"||k=="qwen3.embedding_length")hs=(uint32_t)std::stoul(v);
                else if(k=="qwen3.intermediate_size"||k=="qwen3.feed_forward_length")is_=(uint32_t)std::stoul(v);
                else if(k=="qwen3.block_count")layers=(uint32_t)std::stoul(v);
                else if(k=="qwen3.attention.head_count")heads=(uint32_t)std::stoul(v);
                else if(k=="qwen3.attention.head_count_kv")kv=(uint32_t)std::stoul(v);
                else if(k=="qwen3.vocab_size")vocab=(uint32_t)std::stoul(v);
            }else if(vt==4){uint32_t vu;ru32(vu);
                if(k=="qwen3.hidden_size"||k=="qwen3.embedding_length")hs=vu;
                else if(k=="qwen3.intermediate_size"||k=="qwen3.feed_forward_length")is_=vu;
                else if(k=="qwen3.block_count")layers=vu;
                else if(k=="qwen3.attention.head_count")heads=vu;
                else if(k=="qwen3.attention.head_count_kv")kv=vu;
                else if(k=="qwen3.vocab_size")vocab=vu;
                else if(k=="qwen3.max_position_embeddings")maxseq=vu;
            }else if(vt==6){float vf;f.read((char*)&vf,4);
                if(k=="qwen3.rope.freq_base")rope_theta=vf;
            }else skip(vt);
        }
        for(uint64_t i=0;i<nt;++i){
            GgufTensor t; rstr(t.name); uint32_t nd; ru32(nd);
            t.shape.resize(nd);
            for(uint32_t d=0;d<nd;++d)ru64(t.shape[d]);
            ru32(t.dtype); ru64(t.offset);
            tensors[t.name]=t;
        }
        if(vocab==0){
            auto it=tensors.find("token_embd.weight");
            if(it!=tensors.end()&&it->second.shape.size()>=2)
                vocab=(uint32_t)it->second.shape[1];
        }
        data_start=(uint64_t)f.tellg();
        uint64_t rem=data_start%alignment;
        if(rem)data_start+=alignment-rem;
        return true;
    }

    bool read_f32(const std::string& name, std::vector<float>& out) {
        auto it=tensors.find(name);
        if(it==tensors.end())return false;
        const auto& t=it->second;
        size_t n=1; for(auto d:t.shape)n*=(size_t)d;
        if(t.dtype!=0&&t.dtype!=1)return false;
        f.seekg(data_start+(std::streamoff)t.offset);
        if(t.dtype==0){out.resize(n);f.read((char*)out.data(),n*4);return(size_t)f.gcount()==n*4;}
        std::vector<uint16_t> f16(n); f.read((char*)f16.data(),n*2);
        if((size_t)f.gcount()!=n*2)return false;
        out.resize(n);
        for(size_t i=0;i<n;++i){
            uint16_t h=f16[i]; uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff;
            uint32_t b;
            if(e==0){if(m==0)b=s<<31;else{int e2=-1;uint32_t m2=m;while(!(m2&0x400)){m2<<=1;--e2;}b=(s<<31)|((e2+127)<<23)|((m2&0x3ff)<<13);}}
            else if(e==31)b=(s<<31)|0x7f800000|(m<<13);
            else b=(s<<31)|((e+112)<<23)|(m<<13);
            memcpy(&out[i],&b,4);
        }
        return true;
    }
};

static uint16_t f32_to_f16(float v){
    uint32_t f; memcpy(&f,&v,4);
    uint32_t s=(f>>31)&1; int32_t e=(f>>23)&0xff; uint32_t m=f&0x7fffff;
    if(e==0)return(uint16_t)(s<<15);
    if(e==0xff)return(uint16_t)((s<<15)|0x7c00|(m?0x200:0));
    e=e-127+15; if(e>=31)return(uint16_t)((s<<15)|0x7c00);
    if(e<=0)return(uint16_t)(s<<15);
    return(uint16_t)((s<<15)|(e<<10)|(m>>13));
}

static void pack_tq2_block(const float* vals, uint8_t* out){
    double sa=0; float ma=0;
    for(int i=0;i<128;++i){float a=fabsf(vals[i]);sa+=a;if(a>ma)ma=a;}
    float th=0.7f*(float)(sa/128.0); if(th<1e-10f)th=1e-10f;
    int8_t t[128]; float mm=0;
    for(int i=0;i<128;++i){
        t[i]=(vals[i]>th)?1:((vals[i]<-th)?-1:0);
        if(t[i]!=0){float m=fabsf(vals[i]);if(m>mm)mm=m;}
    }
    if(mm<1e-10f)mm=1.0f;
    uint16_t d=f32_to_f16(mm);
    for(int i=0;i<32;++i){uint8_t b=0;for(int j=0;j<4;++j){int c=t[i*4+j]+1;b|=(uint8_t)(c<<(j*2));}out[i]=b;}
    memcpy(out+32,&d,2);
}

static void ternarize_to_tq2(const float* d, int rows, int cols, std::vector<uint8_t>& out){
    const int gs=128,bb=34,bpr=cols/gs;
    out.resize((size_t)rows*bpr*bb);
    for(int r=0;r<rows;++r)
        for(int b=0;b<bpr;++b)
            pack_tq2_block(d+r*cols+b*gs,out.data()+r*bpr*bb+b*bb);
}

int main(int argc,char**argv){
    const char* in=nullptr,*out=nullptr;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--input"&&i+1<argc)in=argv[++i];
        else if(a=="--output"&&i+1<argc)out=argv[++i];
    }
    if(!in||!out){fprintf(stderr,"usage: gguf_to_h1b --input f16.gguf --output base\n");return 1;}

    GgufReader re;
    if(!re.open(in)){fprintf(stderr,"[err] cannot open %s\n",in);return 1;}
    if(re.arch!="qwen3"){fprintf(stderr,"[err] arch=%s\n",re.arch.c_str());return 1;}
    uint32_t hs=re.hs,is=re.is_,nl=re.layers,nh=re.heads,nkv=re.kv,voc=re.vocab,hd=hs/nh;
    float rt=re.rope_theta;
    fprintf(stderr,"[conv] hs=%u is=%u L=%u nh=%u nkv=%u hd=%u voc=%u\n",hs,is,nl,nh,nkv,hd,voc);
    if(hs==0||is==0||nl==0||voc==0){fprintf(stderr,"[err] bad config\n");return 1;}

    auto rd=[&](const std::string& nm)->std::vector<float>{
        std::vector<float> v; re.read_f32(nm,v);
        if(v.empty())fprintf(stderr,"[warn] missing %s\n",nm.c_str());
        return v;
    };
    auto te=rd("token_embd.weight"),on=rd("output_norm.weight");
    struct N{std::vector<float> an,fn,qn,kn;};
    struct W{std::vector<float> q,k,v,o,g,u,d;};
    std::vector<N> nn(nl);
    std::vector<W> ww(nl);
    for(uint32_t l=0;l<nl;++l){
        std::string p="blk."+std::to_string(l)+".";
        nn[l].an=rd(p+"attn_norm.weight"); nn[l].fn=rd(p+"ffn_norm.weight");
        nn[l].qn=rd(p+"attn_q_norm.weight"); nn[l].kn=rd(p+"attn_k_norm.weight");
        ww[l].q=rd(p+"attn_q.weight"); ww[l].k=rd(p+"attn_k.weight");
        ww[l].v=rd(p+"attn_v.weight"); ww[l].o=rd(p+"attn_output.weight");
        ww[l].g=rd(p+"ffn_gate.weight"); ww[l].u=rd(p+"ffn_up.weight");
        ww[l].d=rd(p+"ffn_down.weight");
    }

    // Write h1b
    std::string h1b=std::string(out)+".h1b";
    fprintf(stderr,"[write] %s\n",h1b.c_str());
    {
        std::ofstream f(h1b,std::ios::binary); if(!f){fprintf(stderr,"[err] write %s\n",h1b.c_str());return 1;}
        f.write("H1B\0",4); int32_t ver=1; f.write((const char*)&ver,4);
        int32_t c9[9]={(int32_t)hs,(int32_t)is,(int32_t)nl,(int32_t)nh,(int32_t)nkv,(int32_t)voc,4096,1,(int32_t)0x8u};
        f.write((const char*)c9,36);
        float ex[2]={rt,1e-5f}; f.write((const char*)ex,8);
        // Embedding zeros
        std::vector<float> zv((size_t)voc*hs,0); f.write((const char*)zv.data(),zv.size()*4);
        std::vector<float> zh(hs,0); f.write((const char*)zh.data(),zh.size()*4);
        // Per-layer norm zeros
        for(uint32_t l=0;l<nl;++l){
            for(int i=0;i<8;++i)f.write((const char*)zh.data(),hs*4);
            std::vector<float> zi(is,0); f.write((const char*)zi.data(),is*4);
        }
        // Packed weights
        auto pw=[&](const std::vector<float>& d,int r,int c){
            if(d.empty()){size_t sz=r*(c/128)*34;std::vector<uint8_t>z(sz,0);f.write((const char*)z.data(),sz);return;}
            std::vector<uint8_t> pk; ternarize_to_tq2(d.data(),r,c,pk); f.write((const char*)pk.data(),pk.size());
        };
        for(uint32_t l=0;l<nl;++l){
            pw(ww[l].q,nh*hd,hs); pw(ww[l].k,nkv*hd,hs); pw(ww[l].v,nkv*hd,hs);
            pw(ww[l].o,hs,nh*hd); pw(ww[l].g,is,hs); pw(ww[l].u,is,hs); pw(ww[l].d,hs,is);
        }
    }

    // Write sidecar GGUF
    std::string gguf=std::string(out)+".gguf";
    fprintf(stderr,"[write] %s\n",gguf.c_str());
    {
        std::ofstream f(gguf,std::ios::binary); if(!f){fprintf(stderr,"[err] write %s\n",gguf.c_str());return 1;}
        auto w32=[&](uint32_t v){f.write((const char*)&v,4);};
        auto w64=[&](uint64_t v){f.write((const char*)&v,8);};
        auto wstr=[&](const std::string& s){w64(s.size());f.write(s.data(),s.size());};
        auto wkvstr=[&](const std::string& k,const std::string& v){wstr(k);w32(8);wstr(v);};
        auto wkvu32=[&](const std::string& k,uint32_t v){wstr(k);w32(4);w32(v);};
        auto wtinfo=[&](const std::string& n,const std::vector<uint64_t>& sh,uint32_t dt,uint64_t off){
            wstr(n);w32((uint32_t)sh.size());for(auto d:sh)w64(d);w32(dt);w64(off);
        };
        auto walign=[&](uint32_t a){
            uint64_t p=(uint64_t)f.tellp(),r=p%a;
            if(r)for(uint64_t i=0;i<a-r;++i)f.put(0);
        };
        uint64_t nt=4*nl+2,nk=9;
        f.write("GGUF",4);w32(3);w64(nt);w64(nk);
        wkvstr("general.architecture","qwen3");
        wkvu32("qwen3.hidden_size",hs); wkvu32("qwen3.feed_forward_length",is);
        wkvu32("qwen3.block_count",nl); wkvu32("qwen3.attention.head_count",nh);
        wkvu32("qwen3.attention.head_count_kv",nkv); wkvu32("qwen3.vocab_size",voc);
        wkvu32("qwen3.max_position_embeddings",4096); wkvu32("qwen3.rope.freq_base",(uint32_t)rt);
        uint64_t off=0;
        for(uint32_t l=0;l<nl;++l){
            wtinfo("blk."+std::to_string(l)+".attn_norm.weight",{hs},0,off);off+=hs*4;
            wtinfo("blk."+std::to_string(l)+".ffn_norm.weight",{hs},0,off);off+=hs*4;
            wtinfo("blk."+std::to_string(l)+".attn_q_norm.weight",{hd},0,off);off+=hd*4;
            wtinfo("blk."+std::to_string(l)+".attn_k_norm.weight",{hd},0,off);off+=hd*4;
        }
        wtinfo("output_norm.weight",{hs},0,off);off+=hs*4;
        wtinfo("token_embd.weight",{hs,voc},42,off); walign(32);
        for(uint32_t l=0;l<nl;++l){
            f.write((const char*)nn[l].an.data(),hs*4);
            f.write((const char*)nn[l].fn.data(),hs*4);
            f.write((const char*)nn[l].qn.data(),hd*4);
            f.write((const char*)nn[l].kn.data(),hd*4);
        }
        f.write((const char*)on.data(),hs*4);
        {
            std::vector<uint8_t> te_packed;
            ternarize_to_tq2(te.data(),(int)voc,(int)hs,te_packed);
            f.write((const char*)te_packed.data(),te_packed.size());
        }
    }
    fprintf(stderr,"[done] %s + %s\n",h1b.c_str(),gguf.c_str());
    return 0;
}
