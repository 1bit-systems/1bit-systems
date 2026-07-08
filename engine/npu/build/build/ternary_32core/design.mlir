module {
  aie.device(npu2) {
    func.func private @mm_ternary_32x64x128(memref<392xi32>, memref<4xf32>, i32, i32) attributes {link_with = "mm_ternary_32x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [])
    aie.objectfifo @A_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_1(%mem_tile_1_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [])
    aie.objectfifo @A_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_2(%mem_tile_2_1, {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [])
    aie.objectfifo @A_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_3(%mem_tile_3_1, {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [])
    aie.objectfifo @A_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_4(%mem_tile_4_1, {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_4] -> [@A_L2L1_4]([] [])
    aie.objectfifo @A_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_5(%mem_tile_5_1, {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_5] -> [@A_L2L1_5]([] [])
    aie.objectfifo @A_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_6(%mem_tile_6_1, {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_6] -> [@A_L2L1_6]([] [])
    aie.objectfifo @A_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo @A_L2L1_7(%mem_tile_7_1, {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<392xi32>> 
    aie.objectfifo.link [@A_L3L2_7] -> [@A_L2L1_7]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 4, 8, 12] [])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<4xf32>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>> 
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 4, 8, 12] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_0(Consume, 1)
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_0(Consume, 1)
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_0(Consume, 1)
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_0(Consume, 1)
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_1(Consume, 1)
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_1(Consume, 1)
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_1(Consume, 1)
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_1(Consume, 1)
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_2(Consume, 1)
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_2(Consume, 1)
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_2(Consume, 1)
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_2(Consume, 1)
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_3(Consume, 1)
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_3(Consume, 1)
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_3(Consume, 1)
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_3(Consume, 1)
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_4(Consume, 1)
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_4(Consume, 1)
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_4(Consume, 1)
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_4(Consume, 1)
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_5(Consume, 1)
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_5(Consume, 1)
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_5(Consume, 1)
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_5(Consume, 1)
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_6(Consume, 1)
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_6(Consume, 1)
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_6(Consume, 1)
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_6(Consume, 1)
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c0_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_7(Consume, 1)
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c4_i32 = arith.constant 4 : i32
        %c4_i32_0 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c4_i32, %c4_i32_0) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_7(Consume, 1)
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c8_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_7(Consume, 1)
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @A_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<392xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<392xi32>> -> memref<392xi32>
        %2 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<4xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4xf32>> -> memref<4xf32>
        %c12_i32 = arith.constant 12 : i32
        %c4_i32 = arith.constant 4 : i32
        func.call @mm_ternary_32x64x128(%1, %3, %c12_i32, %c4_i32) : (memref<392xi32>, memref<4xf32>, i32, i32) -> ()
        aie.objectfifo.release @A_L2L1_7(Consume, 1)
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<3136xi32>, %arg1: memref<128xi32>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 0, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 392, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 784, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 1176, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L2_4 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 1568, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @A_L3L2_5 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 1960, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L2_6 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 2352, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @A_L3L2_7 {
        aie.dma_bd(%arg0 : memref<3136xi32>, 2744, 392, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 392, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg1 : memref<128xi32>, 0, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg1 : memref<128xi32>, 16, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg1 : memref<128xi32>, 32, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg1 : memref<128xi32>, 48, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg1 : memref<128xi32>, 64, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg1 : memref<128xi32>, 80, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg1 : memref<128xi32>, 96, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg1 : memref<128xi32>, 112, 16, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
    }
  }
}

