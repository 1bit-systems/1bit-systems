# Reddit Post: **We ran Qwen3-0.6B on the Strix Halo NPU at 4.8 tok/s -- need help unlocking INT8** : StrixHalo

- **Subreddit:** r/StrixHalo
- **Author:** u/Creepy-Douchebag
- **Score:** 12
- **Comments:** 3
- **Fetched:** 2026-06-29T17:46:32.345082+00:00
- **URL:** https://www.reddit.com/r/StrixHalo/comments/1uitnkr/we_ran_qwen306b_on_the_strix_halo_npu_at_48_toks/

## Post Body

We ran Qwen3-0.6B on the Strix Halo NPU at 4.8 tok/s -- need help unlocking INT8 TL;DR: Reverse-engineered the undocumented AMD XDNA2 NPU on Strix Halo (Ryzen AI MAX+ 395) to run Qwen3-0.6B at 210ms/tok (4.8 tok/s) -- 3.2x faster than CPU. Built everything from scratch -- 15 xclbins, 7 compiler bug fixes, full IRON API pipeline. But INT8 is blocked by the MLIR toolchain (only accepts BFP16 types), and BF16 DMA hangs due to aiecc descriptor generation bugs. Need community help. What We Built The 

## Comments (3)

- **u/RedParaglider** : Who is we? How many are on your team.

- **u/Creepy-Douchebag** : Pi.dev + Deepseek v4 chat

- **u/Poizone360** : Hey, great work. For the INT8 strides, AIE2 DMA appears to address memory in 32-bit words, so this is probably more than just a 1.125 to 1.0 byte change. INT8 likely needs four elements packed per word in the innermost stream dimension, with strides measured in words rather than bytes. It would be worth checking dimensionsToStream against that word alignment, since BFP16’s block layout may be masking the sub-word packing that INT8 now requires. For the BF16 Chess hang, try the Peano/LLVM-AIE bac
