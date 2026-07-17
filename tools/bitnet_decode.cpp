    // For TQ1 (base-3) or Sherry, use the dedicated kernels instead.
    // Block-scaled ternary (BS_T): per-block FP8 scales, no per-row scales tensor.
    const auto ternary_gemv_i8 =
        (m.weight_format == RCPP_WEIGHT_FORMAT_TQ1)                 ? rcpp_ternary_gemv_tq1_halo_f16
      : (m.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_I8)           ? rcpp_ternary_gemv_sherry_f16
      : (m.weight_format == RCPP_WEIGHT_FORMAT_BLOCK_SCALED_TERNARY) ? rcpp_ternary_gemv_bst
                                                                     : rcpp_ternary_gemv_halo_f16;
    // TQ1 needs K multiple of 20 (u32-aligned row bytes).
    const int k_pad_unit = (m.weight_format == RCPP_WEIGHT_FORMAT_TQ1) ? 20 : 1;