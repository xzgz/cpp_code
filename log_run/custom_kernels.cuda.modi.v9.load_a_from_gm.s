	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.protected	_Z20matrixMultiplySharedPfS_S_iiiiii ; -- Begin function _Z20matrixMultiplySharedPfS_S_iiiiii
	.globl	_Z20matrixMultiplySharedPfS_S_iiiiii
	.p2align	8
	.type	_Z20matrixMultiplySharedPfS_S_iiiiii,@function
_Z20matrixMultiplySharedPfS_S_iiiiii:   ; @_Z20matrixMultiplySharedPfS_S_iiiiii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_add_u32 s0, s0, 48
	s_addc_u32 s1, s1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_u32 s15, s17
	s_cselect_b32 s17, 14, 20
	v_mov_b32_e32 v1, s17
	s_cmp_lt_u32 s14, s16
	s_cselect_b32 s16, 12, 18
	global_load_ushort v2, v1, s[0:1]
	v_mov_b32_e32 v1, s16
	global_load_ushort v4, v1, s[0:1]
	v_mov_b32_e32 v1, 0
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v6, 0x3ff, v0
	v_lshlrev_b32_e32 v0, 2, v6
	v_lshlrev_b32_e32 v7, 7, v3
	v_mul_lo_u32 v5, v3, s11
	v_add_u32_e32 v9, v7, v0
	v_or_b32_e32 v10, 0x1000, v0
	v_add_u32_e32 v11, v10, v7
	v_add_u32_e32 v12, 0x400, v10
	v_add_u32_e32 v13, 0x800, v10
	v_add_u32_e32 v14, 0xc00, v10
	s_add_i32 s0, s9, -1
	s_lshl_b32 s20, s11, 5
	s_lshr_b32 s0, s0, 5
	s_add_i32 s21, s0, 1
	s_waitcnt vmcnt(1)
	v_mul_lo_u32 v0, s15, v2
	v_add_u32_e32 v8, v0, v3
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v0, s14, v4
	v_add_u32_e32 v2, v0, v6
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_mul_lo_u32 v15, v8, s9
	v_add3_u32 v4, v6, v5, v0
	v_cmp_gt_i32_e64 s[0:1], s11, v2
	v_mov_b32_e32 v16, 0
	s_branch .LBB0_3
.LBB0_1:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v11, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[34:35], v10 offset1:32
	ds_read_b128 v[18:21], v7
	ds_read_b128 v[22:25], v7 offset:16
	ds_read2_b32 v[36:37], v10 offset0:64 offset1:96
	ds_read_b128 v[26:29], v7 offset:32
	ds_read_b128 v[30:33], v7 offset:48
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v16, v18, v34
	ds_read2_b32 v[38:39], v10 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v35
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v16, v20, v36
	ds_read2_b32 v[18:19], v10 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v37
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v22, v38
	ds_read2_b32 v[20:21], v12 offset1:32
	v_fmac_f32_e32 v16, v23, v39
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v24, v18
	ds_read2_b32 v[22:23], v12 offset0:64 offset1:96
	v_fmac_f32_e32 v16, v25, v19
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v26, v20
	ds_read2_b32 v[18:19], v12 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v27, v21
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v28, v22
	v_fmac_f32_e32 v16, v29, v23
	ds_read2_b32 v[22:23], v12 offset0:192 offset1:224
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v30, v18
	v_fmac_f32_e32 v16, v31, v19
	ds_read2_b32 v[26:27], v13 offset1:32
	ds_read_b128 v[18:21], v7 offset:64
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v32, v22
	v_fmac_f32_e32 v16, v33, v23
	ds_read2_b32 v[28:29], v13 offset0:64 offset1:96
	ds_read_b128 v[22:25], v7 offset:80
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v18, v26
	ds_read2_b32 v[30:31], v13 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v27
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v20, v28
	ds_read2_b32 v[18:19], v13 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v29
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[20:21], v[22:23], v[30:31]
	s_nop 0
	v_add_f32_e32 v0, v16, v20
	v_add_f32_e32 v0, v0, v21
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[20:21], v[24:25], v[18:19]
	ds_read2_b32 v[24:25], v14 offset1:32
	ds_read_b128 v[16:19], v7 offset:96
	v_add_f32_e32 v0, v0, v20
	v_add_f32_e32 v0, v0, v21
	ds_read2_b32 v[26:27], v14 offset0:64 offset1:96
	ds_read_b128 v[20:23], v7 offset:112
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[16:17], v[16:17], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	ds_read2_b32 v[16:17], v14 offset0:128 offset1:160
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[18:19], v[18:19], v[26:27]
	s_nop 0
	v_add_f32_e32 v0, v0, v18
	ds_read2_b32 v[24:25], v14 offset0:192 offset1:224
	v_add_f32_e32 v0, v0, v19
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[16:17], v[20:21], v[16:17]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[16:17], v[22:23], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v16, v0, v17
	s_add_i32 s21, s21, -1
	v_add_u32_e32 v6, 32, v6
	v_add_u32_e32 v4, s20, v4
	s_cmp_eq_u32 s21, 0
	v_add_u32_e32 v3, 32, v3
	s_cbranch_scc1 .LBB0_10
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_7
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s9, v6
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v0, v15, v6
	v_lshl_add_u64 v[18:19], v[0:1], 2, s[2:3]
	global_load_dword v0, v[18:19], off
.LBB0_6:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v9, v0
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s10, v3
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_1
; %bb.9:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v5, v1
	v_lshl_add_u64 v[18:19], v[4:5], 2, s[4:5]
	global_load_dword v0, v[18:19], off
	s_branch .LBB0_1
.LBB0_10:
	v_cmp_gt_i32_e32 vcc, s12, v8
	v_cmp_gt_i32_e64 s[0:1], s13, v2
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_mad_u64_u32 v[0:1], s[0:1], v8, s13, v[2:3]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	global_store_dword v[0:1], v16, off
.LBB0_12:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20matrixMultiplySharedPfS_S_iiiiii
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  12
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 22
		.amdhsa_accum_offset 40
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z20matrixMultiplySharedPfS_S_iiiiii, .Lfunc_end0-_Z20matrixMultiplySharedPfS_S_iiiiii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 908
; NumSgprs: 28
; NumVgprs: 40
; NumAgprs: 0
; TotalNumVgprs: 40
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 8192 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 40
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_cvt_f32_u32_e32 v1, s10
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mul_i32 s12, s12, s10
	s_sub_i32 s0, 0, s10
	v_mul_lo_u32 v2, s0, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_bfe_u32 v2, v0, 10, 10
	v_mul_hi_u32 v1, v2, v1
	v_mul_lo_u32 v1, v1, s10
	v_sub_u32_e32 v1, v2, v1
	v_subrev_u32_e32 v2, s10, v1
	v_cmp_le_u32_e32 vcc, s10, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_subrev_u32_e32 v2, s10, v1
	v_cmp_le_u32_e32 vcc, s10, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_add_u32_e32 v40, s12, v1
	v_cmp_gt_u32_e32 vcc, s3, v40
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_17
; %bb.1:
	s_cmp_lg_u32 s2, 0
	v_and_b32_e32 v0, 0x3ff, v0
	v_lshlrev_b32_e32 v42, 3, v0
	v_cmp_eq_u32_e64 s[0:1], 63, v0
	s_mul_i32 s20, s11, s10
	v_mad_u64_u32 v[44:45], s[10:11], s2, v40, v[42:43]
	s_mul_i32 s21, s20, s2
	v_lshl_add_u32 v43, s2, 1, v42
	v_mad_u64_u32 v[46:47], s[10:11], s2, 3, v[42:43]
	v_add_u32_e32 v45, s2, v42
	s_mov_b64 s[14:15], 0
	s_cselect_b64 s[10:11], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 1, v0
	v_mov_b32_e32 v49, 0
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v40, s20, v40
	v_cmp_le_u32_e32 vcc, s3, v40
	s_or_b64 s[14:15], vcc, s[14:15]
	v_add_u32_e32 v44, s21, v44
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB1_17
.LBB1_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_7 Depth 2
	s_and_b64 vcc, exec, s[10:11]
	s_mov_b32 s22, 0
	s_cbranch_vccnz .LBB1_14
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v41, 0
	s_branch .LBB1_7
.LBB1_5:                                ;   in Loop: Header=BB1_7 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_6:                                ;   in Loop: Header=BB1_7 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s22, 0x400
	s_cmp_ge_u32 s22, s2
	s_cbranch_scc1 .LBB1_15
.LBB1_7:                                ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v52, s22, v42
	v_cmp_gt_u32_e32 vcc, s2, v52
	v_add_u32_e32 v50, 0x200, v52
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_11
; %bb.8:                                ;   in Loop: Header=BB1_7 Depth=2
	v_add_u32_e32 v48, s22, v44
	v_lshl_add_u64 v[4:5], v[48:49], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[4:7], v[4:5], off nt
	;;#ASMEND
	v_mov_b32_e32 v53, v49
	v_lshl_add_u64 v[12:13], v[52:53], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off
	;;#ASMEND
	v_add_u32_e32 v56, s22, v45
	v_mov_b32_e32 v57, v49
	v_lshl_add_u64 v[16:17], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off
	;;#ASMEND
	v_add_u32_e32 v54, s22, v43
	v_mov_b32_e32 v55, v49
	v_lshl_add_u64 v[24:25], v[54:55], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off
	;;#ASMEND
	v_add_u32_e32 v52, s22, v46
	v_lshl_add_u64 v[32:33], v[52:53], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off
	;;#ASMEND
	v_cmp_gt_u32_e64 s[12:13], s2, v50
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_cbranch_execz .LBB1_10
; %bb.9:                                ;   in Loop: Header=BB1_7 Depth=2
	v_add_u32_e32 v48, 0x200, v48
	v_lshl_add_u64 v[0:1], v[48:49], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[0:3], v[0:1], off nt
	;;#ASMEND
	v_mov_b32_e32 v51, v49
	v_lshl_add_u64 v[8:9], v[50:51], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v56
	v_lshl_add_u64 v[20:21], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v54
	v_lshl_add_u64 v[28:29], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v52
	v_lshl_add_u64 v[36:37], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[36:39], v[36:37], off
	;;#ASMEND
.LBB1_10:                               ;   in Loop: Header=BB1_7 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_11:                               ;   in Loop: Header=BB1_7 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB1_6
; %bb.12:                               ;   in Loop: Header=BB1_7 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(5)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v12, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v13, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v14, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v16, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v17, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v18, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v19, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v24, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v25, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v26, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v32, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v33, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v34, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v35, v7
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v50
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_5
; %bb.13:                               ;   in Loop: Header=BB1_7 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v8, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v9, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v11, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v20, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v21, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v22, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v28, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v29, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v30, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v58, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v36, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v37, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v38, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v47, v39, v3
	;;#ASMEND
	s_branch .LBB1_5
.LBB1_14:                               ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b32_e32 v41, v49
	v_mov_b32_e32 v59, v49
	v_mov_b32_e32 v58, v49
	v_mov_b32_e32 v47, v49
.LBB1_15:                               ;   in Loop: Header=BB1_3 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v58, v58, v58 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v47, v47, v47 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB1_2
; %bb.16:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cvt_f16_f32_e32 v48, v41
	v_mov_b32_e32 v41, v49
	v_cvt_f16_f32_e32 v52, v59
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[8:9]
	global_store_short v[50:51], v48, off
	v_add_u32_e32 v48, s3, v40
	v_lshl_add_u64 v[50:51], v[48:49], 1, s[8:9]
	global_store_short v[50:51], v52, off
	v_cvt_f16_f32_e32 v41, v58
	v_add_u32_e32 v48, s3, v48
	v_lshl_add_u64 v[50:51], v[48:49], 1, s[8:9]
	v_cvt_f16_f32_e32 v47, v47
	global_store_short v[50:51], v41, off
	v_add_u32_e32 v48, s3, v48
	v_lshl_add_u64 v[50:51], v[48:49], 1, s[8:9]
	global_store_short v[50:51], v47, off
	s_branch .LBB1_2
.LBB1_17:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 60
		.amdhsa_next_free_sgpr 23
		.amdhsa_accum_offset 60
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end1:
	.size	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end1-_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1556
; NumSgprs: 29
; NumVgprs: 60
; NumAgprs: 0
; TotalNumVgprs: 60
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 60
; AccumOffset: 60
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 14
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,@function
_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii: ; @_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_mul_i32 s12, s12, s10
	v_bfe_u32 v1, v0, 10, 10
	v_add_u32_e32 v40, s12, v1
	v_cmp_gt_u32_e32 vcc, s3, v40
	v_add_u32_e32 v1, 1, v40
	v_cmp_le_u32_e64 s[0:1], s3, v1
	s_and_b64 s[12:13], vcc, s[0:1]
	v_mov_b32_e32 v43, 1
	s_and_saveexec_b64 s[0:1], s[12:13]
; %bb.1:
	v_subrev_u32_e32 v1, s3, v40
	v_cmp_eq_u32_e32 vcc, -1, v1
	s_nop 1
	v_cndmask_b32_e64 v43, 0, 1, vcc
	s_add_i32 s12, s3, -1
	v_mov_b32_e32 v40, s12
; %bb.2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, s3, v40
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_19
; %bb.3:
	s_cmp_lg_u32 s2, 0
	v_and_b32_e32 v0, 0x3ff, v0
	v_lshlrev_b32_e32 v42, 3, v0
	v_cmp_ne_u32_e32 vcc, 63, v0
	s_mul_i32 s20, s11, s10
	s_cselect_b64 s[0:1], -1, 0
	s_add_i32 s21, s3, -1
	s_sub_i32 s22, s20, s3
	s_add_i32 s22, s22, 2
	v_lshl_add_u32 v58, s2, 1, v42
	v_mad_u64_u32 v[44:45], s[10:11], s2, 3, v[42:43]
	v_add_u32_e32 v45, s2, v42
	s_mov_b64 s[14:15], 0
	v_cndmask_b32_e64 v0, 0, 1, s[0:1]
	v_cmp_ne_u32_e64 s[0:1], 1, v0
	s_xor_b64 s[16:17], vcc, -1
	v_mov_b32_e32 v47, 0
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
	s_branch .LBB2_5
.LBB2_4:                                ;   in Loop: Header=BB2_5 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v41, s20, v40
	v_cmp_le_u32_e32 vcc, s3, v41
	v_add_u32_e32 v46, 1, v41
	v_cmp_gt_u32_e64 s[10:11], s3, v46
	v_add_u32_e32 v40, s22, v40
	v_cmp_eq_u32_e64 s[12:13], 1, v40
	v_mov_b32_e32 v40, s21
	s_or_b64 vcc, vcc, s[10:11]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_cmp_le_u32_e64 s[10:11], s3, v40
	s_or_b64 vcc, vcc, s[12:13]
	s_or_b64 s[14:15], s[10:11], s[14:15]
	v_cndmask_b32_e32 v43, 0, v43, vcc
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB2_19
.LBB2_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_9 Depth 2
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s23, 0
	s_cbranch_vccnz .LBB2_16
; %bb.6:                                ;   in Loop: Header=BB2_5 Depth=1
	v_mad_u64_u32 v[48:49], s[10:11], v40, s2, v[42:43]
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v41, 0
	s_branch .LBB2_9
.LBB2_7:                                ;   in Loop: Header=BB2_9 Depth=2
	s_or_b64 exec, exec, s[12:13]
.LBB2_8:                                ;   in Loop: Header=BB2_9 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s23, 0x400
	s_cmp_ge_u32 s23, s2
	s_cbranch_scc1 .LBB2_17
.LBB2_9:                                ;   Parent Loop BB2_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v52, s23, v42
	v_cmp_gt_u32_e32 vcc, s2, v52
	v_add_u32_e32 v50, 0x200, v52
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB2_13
; %bb.10:                               ;   in Loop: Header=BB2_9 Depth=2
	v_add_u32_e32 v46, s23, v48
	v_lshl_add_u64 v[4:5], v[46:47], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[4:7], v[4:5], off nt
	;;#ASMEND
	v_mov_b32_e32 v53, v47
	v_lshl_add_u64 v[12:13], v[52:53], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off
	;;#ASMEND
	v_add_u32_e32 v56, s23, v45
	v_mov_b32_e32 v57, v47
	v_lshl_add_u64 v[16:17], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off
	;;#ASMEND
	v_add_u32_e32 v54, s23, v58
	v_mov_b32_e32 v55, v47
	v_lshl_add_u64 v[24:25], v[54:55], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off
	;;#ASMEND
	v_add_u32_e32 v52, s23, v44
	v_lshl_add_u64 v[32:33], v[52:53], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off
	;;#ASMEND
	v_cmp_gt_u32_e64 s[10:11], s2, v50
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_9 Depth=2
	v_add_u32_e32 v46, 0x200, v46
	v_lshl_add_u64 v[0:1], v[46:47], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[0:3], v[0:1], off nt
	;;#ASMEND
	v_mov_b32_e32 v51, v47
	v_lshl_add_u64 v[8:9], v[50:51], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v56
	v_lshl_add_u64 v[20:21], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v54
	v_lshl_add_u64 v[28:29], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v52
	v_lshl_add_u64 v[36:37], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[36:39], v[36:37], off
	;;#ASMEND
.LBB2_12:                               ;   in Loop: Header=BB2_9 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_13:                               ;   in Loop: Header=BB2_9 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_8
; %bb.14:                               ;   in Loop: Header=BB2_9 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(5)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v12, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v13, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v14, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v15, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v16, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v17, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v18, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v19, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v24, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v25, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v26, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v32, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v33, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v34, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v35, v7
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v50
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB2_7
; %bb.15:                               ;   in Loop: Header=BB2_9 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v8, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v9, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v10, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v41, v11, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v20, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v21, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v22, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v60, v23, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v28, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v29, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v30, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v59, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v36, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v37, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v38, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v49, v39, v3
	;;#ASMEND
	s_branch .LBB2_7
.LBB2_16:                               ;   in Loop: Header=BB2_5 Depth=1
	v_mov_b32_e32 v41, v47
	v_mov_b32_e32 v60, v47
	v_mov_b32_e32 v59, v47
	v_mov_b32_e32 v49, v47
.LBB2_17:                               ;   in Loop: Header=BB2_5 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v41, v41, v41 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v60, v60, v60 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v59, v59, v59 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	v_cmp_ne_u32_e32 vcc, 0, v43
	s_and_b64 s[12:13], s[16:17], vcc
	s_and_saveexec_b64 s[10:11], s[12:13]
	s_cbranch_execz .LBB2_4
; %bb.18:                               ;   in Loop: Header=BB2_5 Depth=1
	v_cvt_f16_f32_e32 v46, v41
	v_mov_b32_e32 v41, v47
	v_cvt_f16_f32_e32 v48, v60
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[8:9]
	global_store_short v[50:51], v46, off
	v_add_u32_e32 v46, s3, v40
	v_lshl_add_u64 v[50:51], v[46:47], 1, s[8:9]
	global_store_short v[50:51], v48, off
	v_cvt_f16_f32_e32 v41, v59
	v_add_u32_e32 v46, s3, v46
	v_lshl_add_u64 v[50:51], v[46:47], 1, s[8:9]
	v_cvt_f16_f32_e32 v52, v49
	global_store_short v[50:51], v41, off
	v_add_u32_e32 v46, s3, v46
	v_lshl_add_u64 v[48:49], v[46:47], 1, s[8:9]
	global_store_short v[48:49], v52, off
	s_branch .LBB2_4
.LBB2_19:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 61
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 64
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end2:
	.size	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end2-_Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1580
; NumSgprs: 30
; NumVgprs: 61
; NumAgprs: 0
; TotalNumVgprs: 61
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 61
; AccumOffset: 64
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 15
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_cvt_f32_u32_e32 v1, s10
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mul_i32 s12, s12, s10
	s_sub_i32 s0, 0, s10
	v_mul_lo_u32 v2, s0, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_bfe_u32 v2, v0, 10, 10
	v_mul_hi_u32 v1, v2, v1
	v_mul_lo_u32 v1, v1, s10
	v_sub_u32_e32 v1, v2, v1
	v_subrev_u32_e32 v2, s10, v1
	v_cmp_le_u32_e32 vcc, s10, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_subrev_u32_e32 v2, s10, v1
	v_cmp_le_u32_e32 vcc, s10, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_add_u32_e32 v40, s12, v1
	v_cmp_gt_u32_e32 vcc, s3, v40
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_33
; %bb.1:
	s_cmp_lg_u32 s2, 0
	v_and_b32_e32 v0, 0x3ff, v0
	v_lshlrev_b32_e32 v42, 3, v0
	v_cmp_eq_u32_e64 s[0:1], 63, v0
	s_mul_i32 s20, s11, s10
	v_mad_u64_u32 v[44:45], s[10:11], s2, v40, v[42:43]
	s_mul_i32 s21, s20, s2
	v_lshl_add_u32 v43, s2, 1, v42
	v_mad_u64_u32 v[46:47], s[10:11], s2, 3, v[42:43]
	v_add_u32_e32 v45, s2, v42
	s_mov_b64 s[14:15], 0
	s_cselect_b64 s[10:11], -1, 0
	v_cndmask_b32_e64 v0, 0, 1, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 1, v0
	s_mov_b32 s22, 0x7f800000
	s_movk_i32 s23, 0x7fff
	v_mov_b32_e32 v49, 0
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
	s_branch .LBB3_4
.LBB3_2:                                ;   in Loop: Header=BB3_4 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v48, s3, v48
	v_lshl_add_u64 v[50:51], v[48:49], 1, s[8:9]
	global_store_short_d16_hi v[50:51], v41, off
.LBB3_3:                                ;   in Loop: Header=BB3_4 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v40, s20, v40
	v_cmp_le_u32_e32 vcc, s3, v40
	s_or_b64 s[14:15], vcc, s[14:15]
	v_add_u32_e32 v44, s21, v44
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB3_33
.LBB3_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_8 Depth 2
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB3_15
; %bb.5:                                ;   in Loop: Header=BB3_4 Depth=1
	s_mov_b32 s24, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v50
	v_mov_b32_e32 v52, v50
	v_mov_b32_e32 v53, v50
	s_branch .LBB3_8
.LBB3_6:                                ;   in Loop: Header=BB3_8 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s24, 0x400
	s_cmp_ge_u32 s24, s2
	s_cbranch_scc1 .LBB3_16
.LBB3_8:                                ;   Parent Loop BB3_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v56, s24, v42
	v_cmp_gt_u32_e32 vcc, s2, v56
	v_add_u32_e32 v54, 0x200, v56
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB3_12
; %bb.9:                                ;   in Loop: Header=BB3_8 Depth=2
	v_add_u32_e32 v48, s24, v44
	v_lshl_add_u64 v[4:5], v[48:49], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[4:7], v[4:5], off nt
	;;#ASMEND
	v_mov_b32_e32 v57, v49
	v_lshl_add_u64 v[12:13], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off
	;;#ASMEND
	v_add_u32_e32 v60, s24, v45
	v_mov_b32_e32 v61, v49
	v_lshl_add_u64 v[16:17], v[60:61], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off
	;;#ASMEND
	v_add_u32_e32 v58, s24, v43
	v_mov_b32_e32 v59, v49
	v_lshl_add_u64 v[24:25], v[58:59], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off
	;;#ASMEND
	v_add_u32_e32 v56, s24, v46
	v_lshl_add_u64 v[32:33], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off
	;;#ASMEND
	v_cmp_gt_u32_e64 s[12:13], s2, v54
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_cbranch_execz .LBB3_11
; %bb.10:                               ;   in Loop: Header=BB3_8 Depth=2
	v_add_u32_e32 v48, 0x200, v48
	v_lshl_add_u64 v[0:1], v[48:49], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[0:3], v[0:1], off nt
	;;#ASMEND
	v_mov_b32_e32 v55, v49
	v_lshl_add_u64 v[8:9], v[54:55], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v60
	v_lshl_add_u64 v[20:21], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v58
	v_lshl_add_u64 v[28:29], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off
	;;#ASMEND
	v_add_u32_e32 v48, 0x200, v56
	v_lshl_add_u64 v[36:37], v[48:49], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[36:39], v[36:37], off
	;;#ASMEND
.LBB3_11:                               ;   in Loop: Header=BB3_8 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB3_12:                               ;   in Loop: Header=BB3_8 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB3_7
; %bb.13:                               ;   in Loop: Header=BB3_8 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(5)
	;;#ASMEND
	v_lshlrev_b32_e32 v48, 16, v4
	v_and_b32_e32 v56, 0xffff0000, v4
	v_lshlrev_b32_e32 v58, 16, v5
	v_and_b32_e32 v60, 0xffff0000, v5
	v_lshlrev_b32_e32 v62, 16, v6
	v_and_b32_e32 v64, 0xffff0000, v6
	v_lshlrev_b32_e32 v66, 16, v7
	v_and_b32_e32 v68, 0xffff0000, v7
	v_lshlrev_b32_e32 v71, 16, v12
	v_lshlrev_b32_e32 v70, 16, v16
	v_pk_mul_f32 v[70:71], v[48:49], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v73, 0xffff0000, v12
	v_and_b32_e32 v72, 0xffff0000, v16
	v_pk_mul_f32 v[72:73], v[56:57], v[72:73] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v75, 16, v13
	v_lshlrev_b32_e32 v74, 16, v17
	v_pk_fma_f32 v[70:71], v[74:75], v[58:59], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v13
	v_and_b32_e32 v74, 0xffff0000, v17
	v_pk_fma_f32 v[72:73], v[74:75], v[60:61], v[72:73] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v75, 16, v14
	v_lshlrev_b32_e32 v74, 16, v18
	v_pk_fma_f32 v[70:71], v[74:75], v[62:63], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v14
	v_and_b32_e32 v74, 0xffff0000, v18
	v_pk_fma_f32 v[72:73], v[74:75], v[64:65], v[72:73] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v75, 16, v15
	v_lshlrev_b32_e32 v74, 16, v19
	v_pk_fma_f32 v[70:71], v[74:75], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v15
	v_and_b32_e32 v74, 0xffff0000, v19
	v_pk_fma_f32 v[72:73], v[74:75], v[68:69], v[72:73] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[70:71], v[70:71], v[72:73]
	s_nop 0
	v_pk_add_f32 v[52:53], v[70:71], v[52:53]
	v_lshlrev_b32_e32 v71, 16, v24
	v_lshlrev_b32_e32 v70, 16, v32
	v_pk_mul_f32 v[70:71], v[48:49], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v73, 0xffff0000, v24
	v_and_b32_e32 v72, 0xffff0000, v32
	v_pk_mul_f32 v[56:57], v[56:57], v[72:73] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v73, 16, v25
	v_lshlrev_b32_e32 v72, 16, v33
	v_pk_fma_f32 v[58:59], v[72:73], v[58:59], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v71, 0xffff0000, v25
	v_and_b32_e32 v70, 0xffff0000, v33
	v_pk_fma_f32 v[56:57], v[70:71], v[60:61], v[56:57] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v61, 16, v26
	v_lshlrev_b32_e32 v60, 16, v34
	v_pk_fma_f32 v[58:59], v[60:61], v[62:63], v[58:59] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v26
	v_and_b32_e32 v60, 0xffff0000, v34
	v_pk_fma_f32 v[56:57], v[60:61], v[64:65], v[56:57] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v61, 16, v27
	v_lshlrev_b32_e32 v60, 16, v35
	v_pk_fma_f32 v[58:59], v[60:61], v[66:67], v[58:59] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v27
	v_and_b32_e32 v60, 0xffff0000, v35
	v_pk_fma_f32 v[56:57], v[60:61], v[68:69], v[56:57] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[56:57], v[58:59], v[56:57]
	s_nop 0
	v_pk_add_f32 v[50:51], v[56:57], v[50:51]
	v_cmp_gt_u32_e32 vcc, s2, v54
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB3_6
; %bb.14:                               ;   in Loop: Header=BB3_8 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_lshlrev_b32_e32 v48, 16, v0
	v_and_b32_e32 v54, 0xffff0000, v0
	v_lshlrev_b32_e32 v56, 16, v1
	v_and_b32_e32 v58, 0xffff0000, v1
	v_lshlrev_b32_e32 v60, 16, v2
	v_and_b32_e32 v62, 0xffff0000, v2
	v_lshlrev_b32_e32 v64, 16, v3
	v_and_b32_e32 v66, 0xffff0000, v3
	v_lshlrev_b32_e32 v69, 16, v8
	v_lshlrev_b32_e32 v68, 16, v20
	v_pk_mul_f32 v[68:69], v[48:49], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v8
	v_and_b32_e32 v70, 0xffff0000, v20
	v_pk_mul_f32 v[70:71], v[54:55], v[70:71] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v73, 16, v9
	v_lshlrev_b32_e32 v72, 16, v21
	v_pk_fma_f32 v[68:69], v[72:73], v[56:57], v[68:69] op_sel_hi:[1,0,1]
	v_and_b32_e32 v73, 0xffff0000, v9
	v_and_b32_e32 v72, 0xffff0000, v21
	v_pk_fma_f32 v[70:71], v[72:73], v[58:59], v[70:71] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v73, 16, v10
	v_lshlrev_b32_e32 v72, 16, v22
	v_pk_fma_f32 v[68:69], v[72:73], v[60:61], v[68:69] op_sel_hi:[1,0,1]
	v_and_b32_e32 v73, 0xffff0000, v10
	v_and_b32_e32 v72, 0xffff0000, v22
	v_pk_fma_f32 v[70:71], v[72:73], v[62:63], v[70:71] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v73, 16, v11
	v_lshlrev_b32_e32 v72, 16, v23
	v_pk_fma_f32 v[68:69], v[72:73], v[64:65], v[68:69] op_sel_hi:[1,0,1]
	v_and_b32_e32 v73, 0xffff0000, v11
	v_and_b32_e32 v72, 0xffff0000, v23
	v_pk_fma_f32 v[70:71], v[72:73], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[68:69], v[68:69], v[70:71]
	s_nop 0
	v_pk_add_f32 v[52:53], v[68:69], v[52:53]
	v_lshlrev_b32_e32 v69, 16, v28
	v_lshlrev_b32_e32 v68, 16, v36
	v_pk_mul_f32 v[68:69], v[48:49], v[68:69] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v28
	v_and_b32_e32 v70, 0xffff0000, v36
	v_pk_mul_f32 v[54:55], v[54:55], v[70:71] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v71, 16, v29
	v_lshlrev_b32_e32 v70, 16, v37
	v_pk_fma_f32 v[56:57], v[70:71], v[56:57], v[68:69] op_sel_hi:[1,0,1]
	v_and_b32_e32 v69, 0xffff0000, v29
	v_and_b32_e32 v68, 0xffff0000, v37
	v_pk_fma_f32 v[54:55], v[68:69], v[58:59], v[54:55] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v59, 16, v30
	v_lshlrev_b32_e32 v58, 16, v38
	v_pk_fma_f32 v[56:57], v[58:59], v[60:61], v[56:57] op_sel_hi:[1,0,1]
	v_and_b32_e32 v59, 0xffff0000, v30
	v_and_b32_e32 v58, 0xffff0000, v38
	v_pk_fma_f32 v[54:55], v[58:59], v[62:63], v[54:55] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v59, 16, v31
	v_lshlrev_b32_e32 v58, 16, v39
	v_pk_fma_f32 v[56:57], v[58:59], v[64:65], v[56:57] op_sel_hi:[1,0,1]
	v_and_b32_e32 v59, 0xffff0000, v31
	v_and_b32_e32 v58, 0xffff0000, v39
	v_pk_fma_f32 v[54:55], v[58:59], v[66:67], v[54:55] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[54:55], v[56:57], v[54:55]
	s_nop 0
	v_pk_add_f32 v[50:51], v[54:55], v[50:51]
	s_branch .LBB3_6
.LBB3_15:                               ;   in Loop: Header=BB3_4 Depth=1
	v_mov_b32_e32 v53, v49
	v_mov_b32_e32 v52, v49
	v_mov_b32_e32 v51, v49
	v_mov_b32_e32 v50, v49
.LBB3_16:                               ;   in Loop: Header=BB3_4 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v53, v53, v53 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v52, v52, v52 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB3_3
; %bb.17:                               ;   in Loop: Header=BB3_4 Depth=1
	v_and_b32_e32 v41, 0x7f800000, v53
	v_cmp_ne_u32_e32 vcc, s22, v41
                                        ; implicit-def: $vgpr47
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
; %bb.18:                               ;   in Loop: Header=BB3_4 Depth=1
	v_bfe_u32 v41, v53, 16, 1
	v_add3_u32 v47, v53, v41, s23
; %bb.19:                               ;   in Loop: Header=BB3_4 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[16:17]
; %bb.20:                               ;   in Loop: Header=BB3_4 Depth=1
	v_or_b32_e32 v41, 0x10000, v53
	v_cmp_eq_u32_sdwa vcc, v53, v49 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v47, v41, v53, vcc
; %bb.21:                               ;   in Loop: Header=BB3_4 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_mov_b32_e32 v41, v49
	v_lshl_add_u64 v[54:55], v[40:41], 1, s[8:9]
	global_store_short_d16_hi v[54:55], v47, off
	v_and_b32_e32 v41, 0x7f800000, v52
	v_cmp_ne_u32_e32 vcc, s22, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
; %bb.22:                               ;   in Loop: Header=BB3_4 Depth=1
	v_bfe_u32 v41, v52, 16, 1
	v_add3_u32 v41, v52, v41, s23
                                        ; implicit-def: $vgpr52
; %bb.23:                               ;   in Loop: Header=BB3_4 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[16:17]
; %bb.24:                               ;   in Loop: Header=BB3_4 Depth=1
	v_or_b32_e32 v41, 0x10000, v52
	v_cmp_eq_u32_sdwa vcc, v52, v49 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v52, vcc
; %bb.25:                               ;   in Loop: Header=BB3_4 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v48, s3, v40
	v_lshl_add_u64 v[52:53], v[48:49], 1, s[8:9]
	global_store_short_d16_hi v[52:53], v41, off
	v_and_b32_e32 v41, 0x7f800000, v51
	v_cmp_ne_u32_e32 vcc, s22, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
; %bb.26:                               ;   in Loop: Header=BB3_4 Depth=1
	v_bfe_u32 v41, v51, 16, 1
	v_add3_u32 v41, v51, v41, s23
; %bb.27:                               ;   in Loop: Header=BB3_4 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[16:17]
; %bb.28:                               ;   in Loop: Header=BB3_4 Depth=1
	v_or_b32_e32 v41, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v49 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v51, vcc
; %bb.29:                               ;   in Loop: Header=BB3_4 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v48, s3, v48
	v_lshl_add_u64 v[52:53], v[48:49], 1, s[8:9]
	global_store_short_d16_hi v[52:53], v41, off
	v_and_b32_e32 v41, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s22, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
; %bb.30:                               ;   in Loop: Header=BB3_4 Depth=1
	v_bfe_u32 v41, v50, 16, 1
	v_add3_u32 v41, v50, v41, s23
                                        ; implicit-def: $vgpr50
; %bb.31:                               ;   in Loop: Header=BB3_4 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[16:17]
	s_cbranch_execz .LBB3_2
; %bb.32:                               ;   in Loop: Header=BB3_4 Depth=1
	v_or_b32_e32 v41, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v49 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v50, vcc
	s_branch .LBB3_2
.LBB3_33:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 76
		.amdhsa_next_free_sgpr 25
		.amdhsa_accum_offset 76
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end3:
	.size	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end3-_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2308
; NumSgprs: 31
; NumVgprs: 76
; NumAgprs: 0
; TotalNumVgprs: 76
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 76
; AccumOffset: 76
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,@function
_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii: ; @_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_mul_i32 s12, s12, s10
	v_bfe_u32 v1, v0, 10, 10
	v_add_u32_e32 v40, s12, v1
	v_cmp_gt_u32_e32 vcc, s3, v40
	v_add_u32_e32 v1, 1, v40
	v_cmp_le_u32_e64 s[0:1], s3, v1
	s_and_b64 s[12:13], vcc, s[0:1]
	v_mov_b32_e32 v43, 1
	s_and_saveexec_b64 s[0:1], s[12:13]
; %bb.1:
	v_subrev_u32_e32 v1, s3, v40
	v_cmp_eq_u32_e32 vcc, -1, v1
	s_nop 1
	v_cndmask_b32_e64 v43, 0, 1, vcc
	s_add_i32 s12, s3, -1
	v_mov_b32_e32 v40, s12
; %bb.2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, s3, v40
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB4_36
; %bb.3:
	s_cmp_lg_u32 s2, 0
	v_and_b32_e32 v0, 0x3ff, v0
	v_lshlrev_b32_e32 v42, 3, v0
	v_cmp_eq_u32_e64 s[0:1], 63, v0
	s_mul_i32 s20, s11, s10
	s_cselect_b64 s[10:11], -1, 0
	s_add_i32 s21, s3, -1
	s_sub_i32 s22, s20, s3
	s_add_i32 s22, s22, 2
	v_lshl_add_u32 v62, s2, 1, v42
	v_mad_u64_u32 v[44:45], s[12:13], s2, 3, v[42:43]
	v_add_u32_e32 v45, s2, v42
	s_mov_b64 s[16:17], 0
	v_cndmask_b32_e64 v0, 0, 1, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 1, v0
	s_mov_b32 s23, 0x7f800000
	s_movk_i32 s24, 0x7fff
	v_mov_b32_e32 v47, 0
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
	s_branch .LBB4_6
.LBB4_4:                                ;   in Loop: Header=BB4_6 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v46, s3, v46
	v_lshl_add_u64 v[48:49], v[46:47], 1, s[8:9]
	global_store_short_d16_hi v[48:49], v41, off
.LBB4_5:                                ;   in Loop: Header=BB4_6 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v41, s20, v40
	v_cmp_le_u32_e32 vcc, s3, v41
	v_add_u32_e32 v46, 1, v41
	v_cmp_gt_u32_e64 s[12:13], s3, v46
	v_add_u32_e32 v40, s22, v40
	v_cmp_eq_u32_e64 s[14:15], 1, v40
	v_mov_b32_e32 v40, s21
	s_or_b64 vcc, vcc, s[12:13]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_cmp_le_u32_e64 s[12:13], s3, v40
	s_or_b64 vcc, vcc, s[14:15]
	s_or_b64 s[16:17], s[12:13], s[16:17]
	v_cndmask_b32_e32 v43, 0, v43, vcc
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB4_36
.LBB4_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_10 Depth 2
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB4_17
; %bb.7:                                ;   in Loop: Header=BB4_6 Depth=1
	v_mad_u64_u32 v[52:53], s[12:13], v40, s2, v[42:43]
	s_mov_b32 s25, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, v48
	v_mov_b32_e32 v50, v48
	v_mov_b32_e32 v51, v48
	s_branch .LBB4_10
.LBB4_8:                                ;   in Loop: Header=BB4_10 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB4_9:                                ;   in Loop: Header=BB4_10 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s25, 0x400
	s_cmp_ge_u32 s25, s2
	s_cbranch_scc1 .LBB4_18
.LBB4_10:                               ;   Parent Loop BB4_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v56, s25, v42
	v_cmp_gt_u32_e32 vcc, s2, v56
	v_add_u32_e32 v54, 0x200, v56
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_14
; %bb.11:                               ;   in Loop: Header=BB4_10 Depth=2
	v_add_u32_e32 v46, s25, v52
	v_lshl_add_u64 v[4:5], v[46:47], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[4:7], v[4:5], off nt
	;;#ASMEND
	v_mov_b32_e32 v57, v47
	v_lshl_add_u64 v[12:13], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off
	;;#ASMEND
	v_add_u32_e32 v60, s25, v45
	v_mov_b32_e32 v61, v47
	v_lshl_add_u64 v[16:17], v[60:61], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off
	;;#ASMEND
	v_add_u32_e32 v58, s25, v62
	v_mov_b32_e32 v59, v47
	v_lshl_add_u64 v[24:25], v[58:59], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off
	;;#ASMEND
	v_add_u32_e32 v56, s25, v44
	v_lshl_add_u64 v[32:33], v[56:57], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off
	;;#ASMEND
	v_cmp_gt_u32_e64 s[12:13], s2, v54
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_cbranch_execz .LBB4_13
; %bb.12:                               ;   in Loop: Header=BB4_10 Depth=2
	v_add_u32_e32 v46, 0x200, v46
	v_lshl_add_u64 v[0:1], v[46:47], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[0:3], v[0:1], off nt
	;;#ASMEND
	v_mov_b32_e32 v55, v47
	v_lshl_add_u64 v[8:9], v[54:55], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v60
	v_lshl_add_u64 v[20:21], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v58
	v_lshl_add_u64 v[28:29], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off
	;;#ASMEND
	v_add_u32_e32 v46, 0x200, v56
	v_lshl_add_u64 v[36:37], v[46:47], 1, s[6:7]
	;;#ASMSTART
	global_load_dwordx4 v[36:39], v[36:37], off
	;;#ASMEND
.LBB4_13:                               ;   in Loop: Header=BB4_10 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB4_14:                               ;   in Loop: Header=BB4_10 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB4_9
; %bb.15:                               ;   in Loop: Header=BB4_10 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(5)
	;;#ASMEND
	v_lshlrev_b32_e32 v46, 16, v4
	v_and_b32_e32 v56, 0xffff0000, v4
	v_lshlrev_b32_e32 v58, 16, v5
	v_and_b32_e32 v60, 0xffff0000, v5
	v_lshlrev_b32_e32 v64, 16, v6
	v_and_b32_e32 v66, 0xffff0000, v6
	v_lshlrev_b32_e32 v68, 16, v7
	v_and_b32_e32 v70, 0xffff0000, v7
	v_lshlrev_b32_e32 v73, 16, v12
	v_lshlrev_b32_e32 v72, 16, v16
	v_pk_mul_f32 v[72:73], v[46:47], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v75, 0xffff0000, v12
	v_and_b32_e32 v74, 0xffff0000, v16
	v_pk_mul_f32 v[74:75], v[56:57], v[74:75] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v77, 16, v13
	v_lshlrev_b32_e32 v76, 16, v17
	v_pk_fma_f32 v[72:73], v[76:77], v[58:59], v[72:73] op_sel_hi:[1,0,1]
	v_and_b32_e32 v77, 0xffff0000, v13
	v_and_b32_e32 v76, 0xffff0000, v17
	v_pk_fma_f32 v[74:75], v[76:77], v[60:61], v[74:75] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v77, 16, v14
	v_lshlrev_b32_e32 v76, 16, v18
	v_pk_fma_f32 v[72:73], v[76:77], v[64:65], v[72:73] op_sel_hi:[1,0,1]
	v_and_b32_e32 v77, 0xffff0000, v14
	v_and_b32_e32 v76, 0xffff0000, v18
	v_pk_fma_f32 v[74:75], v[76:77], v[66:67], v[74:75] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v77, 16, v15
	v_lshlrev_b32_e32 v76, 16, v19
	v_pk_fma_f32 v[72:73], v[76:77], v[68:69], v[72:73] op_sel_hi:[1,0,1]
	v_and_b32_e32 v77, 0xffff0000, v15
	v_and_b32_e32 v76, 0xffff0000, v19
	v_pk_fma_f32 v[74:75], v[76:77], v[70:71], v[74:75] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[72:73], v[72:73], v[74:75]
	s_nop 0
	v_pk_add_f32 v[50:51], v[72:73], v[50:51]
	v_lshlrev_b32_e32 v73, 16, v24
	v_lshlrev_b32_e32 v72, 16, v32
	v_pk_mul_f32 v[72:73], v[46:47], v[72:73] op_sel_hi:[0,1]
	v_and_b32_e32 v75, 0xffff0000, v24
	v_and_b32_e32 v74, 0xffff0000, v32
	v_pk_mul_f32 v[56:57], v[56:57], v[74:75] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v75, 16, v25
	v_lshlrev_b32_e32 v74, 16, v33
	v_pk_fma_f32 v[58:59], v[74:75], v[58:59], v[72:73] op_sel_hi:[1,0,1]
	v_and_b32_e32 v73, 0xffff0000, v25
	v_and_b32_e32 v72, 0xffff0000, v33
	v_pk_fma_f32 v[56:57], v[72:73], v[60:61], v[56:57] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v61, 16, v26
	v_lshlrev_b32_e32 v60, 16, v34
	v_pk_fma_f32 v[58:59], v[60:61], v[64:65], v[58:59] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v26
	v_and_b32_e32 v60, 0xffff0000, v34
	v_pk_fma_f32 v[56:57], v[60:61], v[66:67], v[56:57] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v61, 16, v27
	v_lshlrev_b32_e32 v60, 16, v35
	v_pk_fma_f32 v[58:59], v[60:61], v[68:69], v[58:59] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v27
	v_and_b32_e32 v60, 0xffff0000, v35
	v_pk_fma_f32 v[56:57], v[60:61], v[70:71], v[56:57] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[56:57], v[58:59], v[56:57]
	s_nop 0
	v_pk_add_f32 v[48:49], v[56:57], v[48:49]
	v_cmp_gt_u32_e32 vcc, s2, v54
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_8
; %bb.16:                               ;   in Loop: Header=BB4_10 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_lshlrev_b32_e32 v46, 16, v0
	v_and_b32_e32 v54, 0xffff0000, v0
	v_lshlrev_b32_e32 v56, 16, v1
	v_and_b32_e32 v58, 0xffff0000, v1
	v_lshlrev_b32_e32 v60, 16, v2
	v_and_b32_e32 v64, 0xffff0000, v2
	v_lshlrev_b32_e32 v66, 16, v3
	v_and_b32_e32 v68, 0xffff0000, v3
	v_lshlrev_b32_e32 v71, 16, v8
	v_lshlrev_b32_e32 v70, 16, v20
	v_pk_mul_f32 v[70:71], v[46:47], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v73, 0xffff0000, v8
	v_and_b32_e32 v72, 0xffff0000, v20
	v_pk_mul_f32 v[72:73], v[54:55], v[72:73] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v75, 16, v9
	v_lshlrev_b32_e32 v74, 16, v21
	v_pk_fma_f32 v[70:71], v[74:75], v[56:57], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v9
	v_and_b32_e32 v74, 0xffff0000, v21
	v_pk_fma_f32 v[72:73], v[74:75], v[58:59], v[72:73] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v75, 16, v10
	v_lshlrev_b32_e32 v74, 16, v22
	v_pk_fma_f32 v[70:71], v[74:75], v[60:61], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v10
	v_and_b32_e32 v74, 0xffff0000, v22
	v_pk_fma_f32 v[72:73], v[74:75], v[64:65], v[72:73] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v75, 16, v11
	v_lshlrev_b32_e32 v74, 16, v23
	v_pk_fma_f32 v[70:71], v[74:75], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v75, 0xffff0000, v11
	v_and_b32_e32 v74, 0xffff0000, v23
	v_pk_fma_f32 v[72:73], v[74:75], v[68:69], v[72:73] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[70:71], v[70:71], v[72:73]
	s_nop 0
	v_pk_add_f32 v[50:51], v[70:71], v[50:51]
	v_lshlrev_b32_e32 v71, 16, v28
	v_lshlrev_b32_e32 v70, 16, v36
	v_pk_mul_f32 v[70:71], v[46:47], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v73, 0xffff0000, v28
	v_and_b32_e32 v72, 0xffff0000, v36
	v_pk_mul_f32 v[54:55], v[54:55], v[72:73] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v73, 16, v29
	v_lshlrev_b32_e32 v72, 16, v37
	v_pk_fma_f32 v[56:57], v[72:73], v[56:57], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v71, 0xffff0000, v29
	v_and_b32_e32 v70, 0xffff0000, v37
	v_pk_fma_f32 v[54:55], v[70:71], v[58:59], v[54:55] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v59, 16, v30
	v_lshlrev_b32_e32 v58, 16, v38
	v_pk_fma_f32 v[56:57], v[58:59], v[60:61], v[56:57] op_sel_hi:[1,0,1]
	v_and_b32_e32 v59, 0xffff0000, v30
	v_and_b32_e32 v58, 0xffff0000, v38
	v_pk_fma_f32 v[54:55], v[58:59], v[64:65], v[54:55] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v59, 16, v31
	v_lshlrev_b32_e32 v58, 16, v39
	v_pk_fma_f32 v[56:57], v[58:59], v[66:67], v[56:57] op_sel_hi:[1,0,1]
	v_and_b32_e32 v59, 0xffff0000, v31
	v_and_b32_e32 v58, 0xffff0000, v39
	v_pk_fma_f32 v[54:55], v[58:59], v[68:69], v[54:55] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[54:55], v[56:57], v[54:55]
	s_nop 0
	v_pk_add_f32 v[48:49], v[54:55], v[48:49]
	s_branch .LBB4_8
.LBB4_17:                               ;   in Loop: Header=BB4_6 Depth=1
	v_mov_b32_e32 v51, v47
	v_mov_b32_e32 v50, v47
	v_mov_b32_e32 v49, v47
	v_mov_b32_e32 v48, v47
.LBB4_18:                               ;   in Loop: Header=BB4_6 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v51, v51, v51 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v50, v50, v50 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v49, v49, v49 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v48, v48, v48 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB4_5
; %bb.19:                               ;   in Loop: Header=BB4_6 Depth=1
	v_cmp_ne_u32_e32 vcc, 0, v43
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB4_5
; %bb.20:                               ;   in Loop: Header=BB4_6 Depth=1
	v_and_b32_e32 v41, 0x7f800000, v51
	v_cmp_ne_u32_e32 vcc, s23, v41
                                        ; implicit-def: $vgpr46
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.21:                               ;   in Loop: Header=BB4_6 Depth=1
	v_bfe_u32 v41, v51, 16, 1
	v_add3_u32 v46, v51, v41, s24
; %bb.22:                               ;   in Loop: Header=BB4_6 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.23:                               ;   in Loop: Header=BB4_6 Depth=1
	v_or_b32_e32 v41, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v47 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v46, v41, v51, vcc
; %bb.24:                               ;   in Loop: Header=BB4_6 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v41, v47
	v_lshl_add_u64 v[52:53], v[40:41], 1, s[8:9]
	global_store_short_d16_hi v[52:53], v46, off
	v_and_b32_e32 v41, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s23, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.25:                               ;   in Loop: Header=BB4_6 Depth=1
	v_bfe_u32 v41, v50, 16, 1
	v_add3_u32 v41, v50, v41, s24
                                        ; implicit-def: $vgpr50
; %bb.26:                               ;   in Loop: Header=BB4_6 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.27:                               ;   in Loop: Header=BB4_6 Depth=1
	v_or_b32_e32 v41, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v47 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v50, vcc
; %bb.28:                               ;   in Loop: Header=BB4_6 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v46, s3, v40
	v_lshl_add_u64 v[50:51], v[46:47], 1, s[8:9]
	global_store_short_d16_hi v[50:51], v41, off
	v_and_b32_e32 v41, 0x7f800000, v49
	v_cmp_ne_u32_e32 vcc, s23, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.29:                               ;   in Loop: Header=BB4_6 Depth=1
	v_bfe_u32 v41, v49, 16, 1
	v_add3_u32 v41, v49, v41, s24
; %bb.30:                               ;   in Loop: Header=BB4_6 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.31:                               ;   in Loop: Header=BB4_6 Depth=1
	v_or_b32_e32 v41, 0x10000, v49
	v_cmp_eq_u32_sdwa vcc, v49, v47 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v49, vcc
; %bb.32:                               ;   in Loop: Header=BB4_6 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v46, s3, v46
	v_lshl_add_u64 v[50:51], v[46:47], 1, s[8:9]
	global_store_short_d16_hi v[50:51], v41, off
	v_and_b32_e32 v41, 0x7f800000, v48
	v_cmp_ne_u32_e32 vcc, s23, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.33:                               ;   in Loop: Header=BB4_6 Depth=1
	v_bfe_u32 v41, v48, 16, 1
	v_add3_u32 v41, v48, v41, s24
                                        ; implicit-def: $vgpr48
; %bb.34:                               ;   in Loop: Header=BB4_6 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB4_4
; %bb.35:                               ;   in Loop: Header=BB4_6 Depth=1
	v_or_b32_e32 v41, 0x10000, v48
	v_cmp_eq_u32_sdwa vcc, v48, v47 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v48, vcc
	s_branch .LBB4_4
.LBB4_36:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 78
		.amdhsa_next_free_sgpr 26
		.amdhsa_accum_offset 80
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end4:
	.size	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end4-_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2336
; NumSgprs: 32
; NumVgprs: 78
; NumAgprs: 0
; TotalNumVgprs: 78
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 78
; AccumOffset: 80
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 19
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_c3a285be362c239c,@object ; @__hip_cuid_c3a285be362c239c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c3a285be362c239c
__hip_cuid_c3a285be362c239c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c3a285be362c239c, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c3a285be362c239c
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20matrixMultiplySharedPfS_S_iiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z20matrixMultiplySharedPfS_S_iiiiii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     40
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     60
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z12wvSplitK_hf_I6__halfLi64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     61
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     76
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi1ELi8ELi2ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     78
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
