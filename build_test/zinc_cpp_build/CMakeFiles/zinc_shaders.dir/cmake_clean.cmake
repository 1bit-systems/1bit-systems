file(REMOVE_RECURSE
  "CMakeFiles/zinc_shaders"
  "shaders/argmax.spv"
  "shaders/copy_buffer.spv"
  "shaders/dmmv_q4k.spv"
  "shaders/embed.spv"
  "shaders/flash_attn.spv"
  "shaders/fused_gate_up.spv"
  "shaders/fused_qkv.spv"
  "shaders/fused_silu_down.spv"
  "shaders/gemv_f32.spv"
  "shaders/rms_norm.spv"
  "shaders/rms_norm_mul.spv"
  "shaders/rope_fused.spv"
  "shaders/silu_mul.spv"
  "shaders/swiglu.spv"
  "shaders/vadd.spv"
  "shaders/vadd_f32.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/zinc_shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
