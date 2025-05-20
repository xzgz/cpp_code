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
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v80, 3, v2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_lshl_b32 s0, s2, 2
	s_min_i32 s20, s0, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v80
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_5
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s20, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s20
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB1_2
.LBB1_5:
	s_or_b64 exec, exec, s[0:1]
.LBB1_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_32
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v82, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v82
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_32
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s22, s11, s10
	v_mad_u64_u32 v[84:85], s[6:7], s2, v82, v[80:81]
	s_mul_i32 s23, s22, s2
	v_lshl_add_u32 v81, s2, 1, v80
	v_mad_u64_u32 v[86:87], s[6:7], s2, 3, v[80:81]
	v_add_u32_e32 v85, s2, v80
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v87, 0x400
	v_mov_b32_e32 v90, 0x800
	v_mov_b32_e32 v91, 0xc00
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr52_vgpr53_vgpr54_vgpr55
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
                                        ; implicit-def: $vgpr48_vgpr49_vgpr50_vgpr51
                                        ; implicit-def: $vgpr56_vgpr57_vgpr58_vgpr59
                                        ; implicit-def: $vgpr64_vgpr65_vgpr66_vgpr67
                                        ; implicit-def: $vgpr60_vgpr61_vgpr62_vgpr63
                                        ; implicit-def: $vgpr68_vgpr69_vgpr70_vgpr71
                                        ; implicit-def: $vgpr72_vgpr73_vgpr74_vgpr75
                                        ; implicit-def: $vgpr76_vgpr77_vgpr78_vgpr79
	s_branch .LBB1_10
.LBB1_9:                                ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v82, s22, v82
	v_cmp_le_u32_e32 vcc, s3, v82
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v84, s23, v84
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB1_32
.LBB1_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_16 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_mov_b32 s24, 0
	s_cbranch_vccnz .LBB1_29
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v83, 0
	s_branch .LBB1_16
.LBB1_12:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_13:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_14:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB1_15:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s24, 0x800
	s_cmp_ge_u32 s24, s2
	s_cbranch_scc1 .LBB1_30
.LBB1_16:                               ;   Parent Loop BB1_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v95, s24, v80
	v_cmp_gt_u32_e32 vcc, s2, v95
	v_add_u32_e32 v96, 0x200, v95
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_24
; %bb.17:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v88, s24, v84
	v_lshl_add_u64 v[28:29], v[88:89], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[76:79], v[28:29], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v28, 1, v95
	;;#ASMSTART
	ds_read_b128 v[64:67], v28
	;;#ASMEND
	v_add_u32_e32 v97, s24, v85
	v_lshlrev_b32_e32 v28, 1, v97
	;;#ASMSTART
	ds_read_b128 v[52:55], v28
	;;#ASMEND
	v_add_u32_e32 v98, s24, v81
	v_lshlrev_b32_e32 v28, 1, v98
	;;#ASMSTART
	ds_read_b128 v[40:43], v28
	;;#ASMEND
	v_add_u32_e32 v99, s24, v86
	v_lshlrev_b32_e32 v28, 1, v99
	;;#ASMSTART
	ds_read_b128 v[28:31], v28
	;;#ASMEND
	v_cmp_gt_u32_e64 s[10:11], s2, v96
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB1_23
; %bb.18:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v4, 0x200, v88
	v_mov_b32_e32 v5, v89
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[72:75], v[4:5], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v4, 1, v96
	;;#ASMSTART
	ds_read_b128 v[56:59], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v97, 1, v87
	;;#ASMSTART
	ds_read_b128 v[36:39], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v98, 1, v87
	;;#ASMSTART
	ds_read_b128 v[16:19], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v99, 1, v87
	;;#ASMSTART
	ds_read_b128 v[4:7], v4
	;;#ASMEND
	v_add_u32_e32 v100, 0x400, v95
	v_cmp_gt_u32_e64 s[10:11], s2, v100
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB1_22
; %bb.19:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v0, 0x400, v88
	v_mov_b32_e32 v1, v89
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[68:71], v[0:1], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v0, 1, v100
	;;#ASMSTART
	ds_read_b128 v[48:51], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v97, 1, v90
	;;#ASMSTART
	ds_read_b128 v[24:27], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v98, 1, v90
	;;#ASMSTART
	ds_read_b128 v[8:11], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v99, 1, v90
	;;#ASMSTART
	ds_read_b128 v[0:3], v0
	;;#ASMEND
	v_add_u32_e32 v100, 0x600, v95
	v_cmp_gt_u32_e64 s[10:11], s2, v100
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB1_21
; %bb.20:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v88, 0x600, v88
	v_lshl_add_u64 v[12:13], v[88:89], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[60:63], v[12:13], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v12, 1, v100
	;;#ASMSTART
	ds_read_b128 v[44:47], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v97, 1, v91
	;;#ASMSTART
	ds_read_b128 v[32:35], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v98, 1, v91
	;;#ASMSTART
	ds_read_b128 v[20:23], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v99, 1, v91
	;;#ASMSTART
	ds_read_b128 v[12:15], v12
	;;#ASMEND
.LBB1_21:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB1_22:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_23:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_24:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_15
; %bb.25:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(3)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v64, v76
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v65, v77
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v66, v78
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v67, v79
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v52, v76
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v53, v77
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v54, v78
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v55, v79
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v40, v76
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v41, v77
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v42, v78
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v43, v79
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v28, v76
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v29, v77
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v30, v78
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v31, v79
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v96
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_14
; %bb.26:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v56, v72
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v57, v73
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v58, v74
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v59, v75
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v36, v72
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v37, v73
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v38, v74
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v39, v75
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v16, v72
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v17, v73
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v18, v74
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v19, v75
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v4, v72
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v5, v73
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v6, v74
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v7, v75
	;;#ASMEND
	v_add_u32_e32 v88, 0x400, v95
	v_cmp_gt_u32_e32 vcc, s2, v88
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_13
; %bb.27:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(1)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v48, v68
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v49, v69
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v50, v70
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v51, v71
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v24, v68
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v25, v69
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v26, v70
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v27, v71
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v8, v68
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v9, v69
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v10, v70
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v11, v71
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v0, v68
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v1, v69
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v2, v70
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v3, v71
	;;#ASMEND
	v_add_u32_e32 v88, 0x600, v95
	v_cmp_gt_u32_e32 vcc, s2, v88
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_12
; %bb.28:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v44, v60
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v45, v61
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v46, v62
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v83, v47, v63
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v32, v60
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v33, v61
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v34, v62
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v94, v35, v63
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v20, v60
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v21, v61
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v22, v62
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v93, v23, v63
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v12, v60
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v13, v61
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v14, v62
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v92, v15, v63
	;;#ASMEND
	s_branch .LBB1_12
.LBB1_29:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v83, v89
	v_mov_b32_e32 v94, v89
	v_mov_b32_e32 v93, v89
	v_mov_b32_e32 v92, v89
.LBB1_30:                               ;   in Loop: Header=BB1_10 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v83, v83, v83 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v94, v94, v94 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB1_9
; %bb.31:                               ;   in Loop: Header=BB1_10 Depth=1
	v_cvt_f16_f32_e32 v88, v83
	v_mov_b32_e32 v83, v89
	v_cvt_f16_f32_e32 v96, v94
	v_lshl_add_u64 v[94:95], v[82:83], 1, s[8:9]
	global_store_short v[94:95], v88, off
	v_add_u32_e32 v88, s3, v82
	v_lshl_add_u64 v[94:95], v[88:89], 1, s[8:9]
	global_store_short v[94:95], v96, off
	v_cvt_f16_f32_e32 v83, v93
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[94:95], v[88:89], 1, s[8:9]
	v_cvt_f16_f32_e32 v96, v92
	global_store_short v[94:95], v83, off
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[92:93], v[88:89], 1, s[8:9]
	global_store_short v[92:93], v96, off
	s_branch .LBB1_9
.LBB1_32:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
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
		.amdhsa_next_free_vgpr 101
		.amdhsa_next_free_sgpr 25
		.amdhsa_accum_offset 104
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
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end1:
	.size	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end1-_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2284
; NumSgprs: 31
; NumVgprs: 101
; NumAgprs: 0
; TotalNumVgprs: 101
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 101
; AccumOffset: 104
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 25
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v80, 3, v2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB2_6
; %bb.1:
	s_lshl_b32 s0, s2, 2
	s_min_i32 s20, s0, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v80
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB2_3
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s20, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s20
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB2_2
.LBB2_5:
	s_or_b64 exec, exec, s[0:1]
.LBB2_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_48
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v82, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v82
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_48
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s22, s11, s10
	v_mad_u64_u32 v[84:85], s[6:7], s2, v82, v[80:81]
	s_mul_i32 s23, s22, s2
	v_lshl_add_u32 v81, s2, 1, v80
	v_mad_u64_u32 v[86:87], s[6:7], s2, 3, v[80:81]
	v_add_u32_e32 v85, s2, v80
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v89, 0
	s_mov_b32 s24, 0x7f800000
	s_movk_i32 s25, 0x7fff
	v_mov_b32_e32 v87, 0x400
	v_mov_b32_e32 v94, 0x800
	v_mov_b32_e32 v95, 0xc00
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr52_vgpr53_vgpr54_vgpr55
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
                                        ; implicit-def: $vgpr48_vgpr49_vgpr50_vgpr51
                                        ; implicit-def: $vgpr56_vgpr57_vgpr58_vgpr59
                                        ; implicit-def: $vgpr64_vgpr65_vgpr66_vgpr67
                                        ; implicit-def: $vgpr60_vgpr61_vgpr62_vgpr63
                                        ; implicit-def: $vgpr68_vgpr69_vgpr70_vgpr71
                                        ; implicit-def: $vgpr72_vgpr73_vgpr74_vgpr75
                                        ; implicit-def: $vgpr76_vgpr77_vgpr78_vgpr79
	s_branch .LBB2_11
.LBB2_9:                                ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[90:91], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[90:91], v83, off
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v82, s22, v82
	v_cmp_le_u32_e32 vcc, s3, v82
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v84, s23, v84
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB2_48
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_17 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB2_30
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	s_mov_b32 s26, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, v90
	v_mov_b32_e32 v92, v90
	v_mov_b32_e32 v93, v90
	s_branch .LBB2_17
.LBB2_13:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_14:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB2_15:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB2_16:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s26, 0x800
	s_cmp_ge_u32 s26, s2
	s_cbranch_scc1 .LBB2_31
.LBB2_17:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v83, s26, v80
	v_cmp_gt_u32_e32 vcc, s2, v83
	v_add_u32_e32 v96, 0x200, v83
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_25
; %bb.18:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v88, s26, v84
	v_lshl_add_u64 v[28:29], v[88:89], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[76:79], v[28:29], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v28, 1, v83
	;;#ASMSTART
	ds_read_b128 v[64:67], v28
	;;#ASMEND
	v_add_u32_e32 v97, s26, v85
	v_lshlrev_b32_e32 v28, 1, v97
	;;#ASMSTART
	ds_read_b128 v[52:55], v28
	;;#ASMEND
	v_add_u32_e32 v98, s26, v81
	v_lshlrev_b32_e32 v28, 1, v98
	;;#ASMSTART
	ds_read_b128 v[40:43], v28
	;;#ASMEND
	v_add_u32_e32 v99, s26, v86
	v_lshlrev_b32_e32 v28, 1, v99
	;;#ASMSTART
	ds_read_b128 v[28:31], v28
	;;#ASMEND
	v_cmp_gt_u32_e64 s[10:11], s2, v96
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB2_24
; %bb.19:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v4, 0x200, v88
	v_mov_b32_e32 v5, v89
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[72:75], v[4:5], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v4, 1, v96
	;;#ASMSTART
	ds_read_b128 v[56:59], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v97, 1, v87
	;;#ASMSTART
	ds_read_b128 v[36:39], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v98, 1, v87
	;;#ASMSTART
	ds_read_b128 v[16:19], v4
	;;#ASMEND
	v_lshl_add_u32 v4, v99, 1, v87
	;;#ASMSTART
	ds_read_b128 v[4:7], v4
	;;#ASMEND
	v_add_u32_e32 v100, 0x400, v83
	v_cmp_gt_u32_e64 s[10:11], s2, v100
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_23
; %bb.20:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v0, 0x400, v88
	v_mov_b32_e32 v1, v89
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[68:71], v[0:1], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v0, 1, v100
	;;#ASMSTART
	ds_read_b128 v[48:51], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v97, 1, v94
	;;#ASMSTART
	ds_read_b128 v[24:27], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v98, 1, v94
	;;#ASMSTART
	ds_read_b128 v[8:11], v0
	;;#ASMEND
	v_lshl_add_u32 v0, v99, 1, v94
	;;#ASMSTART
	ds_read_b128 v[0:3], v0
	;;#ASMEND
	v_add_u32_e32 v100, 0x600, v83
	v_cmp_gt_u32_e64 s[10:11], s2, v100
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB2_22
; %bb.21:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v88, 0x600, v88
	v_lshl_add_u64 v[12:13], v[88:89], 1, s[4:5]
	;;#ASMSTART
	global_load_dwordx4 v[60:63], v[12:13], off nt
	;;#ASMEND
	v_lshlrev_b32_e32 v12, 1, v100
	;;#ASMSTART
	ds_read_b128 v[44:47], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v97, 1, v95
	;;#ASMSTART
	ds_read_b128 v[32:35], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v98, 1, v95
	;;#ASMSTART
	ds_read_b128 v[20:23], v12
	;;#ASMEND
	v_lshl_add_u32 v12, v99, 1, v95
	;;#ASMSTART
	ds_read_b128 v[12:15], v12
	;;#ASMEND
.LBB2_22:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_23:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_24:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB2_25:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_16
; %bb.26:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(3)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	v_lshlrev_b32_e32 v88, 16, v76
	v_and_b32_e32 v98, 0xffff0000, v76
	v_lshlrev_b32_e32 v101, 16, v64
	v_and_b32_e32 v103, 0xffff0000, v64
	v_lshlrev_b32_e32 v104, 16, v77
	v_and_b32_e32 v106, 0xffff0000, v77
	v_lshlrev_b32_e32 v108, 16, v78
	v_and_b32_e32 v110, 0xffff0000, v78
	v_lshlrev_b32_e32 v112, 16, v79
	v_and_b32_e32 v114, 0xffff0000, v79
	v_lshlrev_b32_e32 v100, 16, v52
	v_and_b32_e32 v102, 0xffff0000, v52
	v_pk_mul_f32 v[100:101], v[88:89], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[98:99], v[102:103] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v117, 16, v65
	v_lshlrev_b32_e32 v116, 16, v53
	v_pk_fma_f32 v[100:101], v[116:117], v[104:105], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v117, 0xffff0000, v65
	v_and_b32_e32 v116, 0xffff0000, v53
	v_pk_fma_f32 v[102:103], v[116:117], v[106:107], v[102:103] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v117, 16, v66
	v_lshlrev_b32_e32 v116, 16, v54
	v_pk_fma_f32 v[100:101], v[116:117], v[108:109], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v117, 0xffff0000, v66
	v_and_b32_e32 v116, 0xffff0000, v54
	v_pk_fma_f32 v[102:103], v[116:117], v[110:111], v[102:103] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v117, 16, v67
	v_lshlrev_b32_e32 v116, 16, v55
	v_pk_fma_f32 v[100:101], v[116:117], v[112:113], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v117, 0xffff0000, v67
	v_and_b32_e32 v116, 0xffff0000, v55
	v_pk_fma_f32 v[102:103], v[116:117], v[114:115], v[102:103] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[100:101], v[100:101], v[102:103]
	s_nop 0
	v_pk_add_f32 v[92:93], v[100:101], v[92:93]
	v_lshlrev_b32_e32 v101, 16, v40
	v_and_b32_e32 v103, 0xffff0000, v40
	v_lshlrev_b32_e32 v100, 16, v28
	v_and_b32_e32 v102, 0xffff0000, v28
	v_pk_mul_f32 v[100:101], v[88:89], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[98:99], v[102:103] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v103, 16, v41
	v_lshlrev_b32_e32 v102, 16, v29
	v_pk_fma_f32 v[100:101], v[102:103], v[104:105], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v103, 0xffff0000, v41
	v_and_b32_e32 v102, 0xffff0000, v29
	v_pk_fma_f32 v[98:99], v[102:103], v[106:107], v[98:99] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v103, 16, v42
	v_lshlrev_b32_e32 v102, 16, v30
	v_pk_fma_f32 v[100:101], v[102:103], v[108:109], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v103, 0xffff0000, v42
	v_and_b32_e32 v102, 0xffff0000, v30
	v_pk_fma_f32 v[98:99], v[102:103], v[110:111], v[98:99] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v103, 16, v43
	v_lshlrev_b32_e32 v102, 16, v31
	v_pk_fma_f32 v[100:101], v[102:103], v[112:113], v[100:101] op_sel_hi:[1,0,1]
	v_and_b32_e32 v103, 0xffff0000, v43
	v_and_b32_e32 v102, 0xffff0000, v31
	v_pk_fma_f32 v[98:99], v[102:103], v[114:115], v[98:99] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[98:99], v[100:101], v[98:99]
	s_nop 0
	v_pk_add_f32 v[90:91], v[98:99], v[90:91]
	v_cmp_gt_u32_e32 vcc, s2, v96
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_15
; %bb.27:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	v_lshlrev_b32_e32 v88, 16, v72
	v_and_b32_e32 v96, 0xffff0000, v72
	v_lshlrev_b32_e32 v98, 16, v73
	v_and_b32_e32 v100, 0xffff0000, v73
	v_lshlrev_b32_e32 v102, 16, v74
	v_and_b32_e32 v104, 0xffff0000, v74
	v_lshlrev_b32_e32 v106, 16, v75
	v_and_b32_e32 v108, 0xffff0000, v75
	v_lshlrev_b32_e32 v111, 16, v56
	v_lshlrev_b32_e32 v110, 16, v36
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v56
	v_and_b32_e32 v112, 0xffff0000, v36
	v_pk_mul_f32 v[112:113], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v115, 16, v57
	v_lshlrev_b32_e32 v114, 16, v37
	v_pk_fma_f32 v[110:111], v[114:115], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v57
	v_and_b32_e32 v114, 0xffff0000, v37
	v_pk_fma_f32 v[112:113], v[114:115], v[100:101], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v58
	v_lshlrev_b32_e32 v114, 16, v38
	v_pk_fma_f32 v[110:111], v[114:115], v[102:103], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v58
	v_and_b32_e32 v114, 0xffff0000, v38
	v_pk_fma_f32 v[112:113], v[114:115], v[104:105], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v59
	v_lshlrev_b32_e32 v114, 16, v39
	v_pk_fma_f32 v[110:111], v[114:115], v[106:107], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v59
	v_and_b32_e32 v114, 0xffff0000, v39
	v_pk_fma_f32 v[112:113], v[114:115], v[108:109], v[112:113] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[110:111], v[110:111], v[112:113]
	s_nop 0
	v_pk_add_f32 v[92:93], v[110:111], v[92:93]
	v_lshlrev_b32_e32 v111, 16, v16
	v_lshlrev_b32_e32 v110, 16, v4
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v16
	v_and_b32_e32 v112, 0xffff0000, v4
	v_pk_mul_f32 v[96:97], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v113, 16, v17
	v_lshlrev_b32_e32 v112, 16, v5
	v_pk_fma_f32 v[98:99], v[112:113], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v111, 0xffff0000, v17
	v_and_b32_e32 v110, 0xffff0000, v5
	v_pk_fma_f32 v[96:97], v[110:111], v[100:101], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v18
	v_lshlrev_b32_e32 v100, 16, v6
	v_pk_fma_f32 v[98:99], v[100:101], v[102:103], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v18
	v_and_b32_e32 v100, 0xffff0000, v6
	v_pk_fma_f32 v[96:97], v[100:101], v[104:105], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v19
	v_lshlrev_b32_e32 v100, 16, v7
	v_pk_fma_f32 v[98:99], v[100:101], v[106:107], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v19
	v_and_b32_e32 v100, 0xffff0000, v7
	v_pk_fma_f32 v[96:97], v[100:101], v[108:109], v[96:97] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[96:97], v[98:99], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[96:97], v[90:91]
	v_add_u32_e32 v88, 0x400, v83
	v_cmp_gt_u32_e32 vcc, s2, v88
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_14
; %bb.28:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(1)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	v_lshlrev_b32_e32 v88, 16, v68
	v_and_b32_e32 v96, 0xffff0000, v68
	v_lshlrev_b32_e32 v98, 16, v69
	v_and_b32_e32 v100, 0xffff0000, v69
	v_lshlrev_b32_e32 v102, 16, v70
	v_and_b32_e32 v104, 0xffff0000, v70
	v_lshlrev_b32_e32 v106, 16, v71
	v_and_b32_e32 v108, 0xffff0000, v71
	v_lshlrev_b32_e32 v111, 16, v48
	v_lshlrev_b32_e32 v110, 16, v24
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v48
	v_and_b32_e32 v112, 0xffff0000, v24
	v_pk_mul_f32 v[112:113], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v115, 16, v49
	v_lshlrev_b32_e32 v114, 16, v25
	v_pk_fma_f32 v[110:111], v[114:115], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v49
	v_and_b32_e32 v114, 0xffff0000, v25
	v_pk_fma_f32 v[112:113], v[114:115], v[100:101], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v50
	v_lshlrev_b32_e32 v114, 16, v26
	v_pk_fma_f32 v[110:111], v[114:115], v[102:103], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v50
	v_and_b32_e32 v114, 0xffff0000, v26
	v_pk_fma_f32 v[112:113], v[114:115], v[104:105], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v51
	v_lshlrev_b32_e32 v114, 16, v27
	v_pk_fma_f32 v[110:111], v[114:115], v[106:107], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v51
	v_and_b32_e32 v114, 0xffff0000, v27
	v_pk_fma_f32 v[112:113], v[114:115], v[108:109], v[112:113] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[110:111], v[110:111], v[112:113]
	s_nop 0
	v_pk_add_f32 v[92:93], v[110:111], v[92:93]
	v_lshlrev_b32_e32 v111, 16, v8
	v_lshlrev_b32_e32 v110, 16, v0
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v8
	v_and_b32_e32 v112, 0xffff0000, v0
	v_pk_mul_f32 v[96:97], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v113, 16, v9
	v_lshlrev_b32_e32 v112, 16, v1
	v_pk_fma_f32 v[98:99], v[112:113], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v111, 0xffff0000, v9
	v_and_b32_e32 v110, 0xffff0000, v1
	v_pk_fma_f32 v[96:97], v[110:111], v[100:101], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v10
	v_lshlrev_b32_e32 v100, 16, v2
	v_pk_fma_f32 v[98:99], v[100:101], v[102:103], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v10
	v_and_b32_e32 v100, 0xffff0000, v2
	v_pk_fma_f32 v[96:97], v[100:101], v[104:105], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v11
	v_lshlrev_b32_e32 v100, 16, v3
	v_pk_fma_f32 v[98:99], v[100:101], v[106:107], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v11
	v_and_b32_e32 v100, 0xffff0000, v3
	v_pk_fma_f32 v[96:97], v[100:101], v[108:109], v[96:97] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[96:97], v[98:99], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[96:97], v[90:91]
	v_add_u32_e32 v83, 0x600, v83
	v_cmp_gt_u32_e32 vcc, s2, v83
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_13
; %bb.29:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_lshlrev_b32_e32 v88, 16, v60
	v_and_b32_e32 v96, 0xffff0000, v60
	v_lshlrev_b32_e32 v98, 16, v61
	v_and_b32_e32 v100, 0xffff0000, v61
	v_lshlrev_b32_e32 v102, 16, v62
	v_and_b32_e32 v104, 0xffff0000, v62
	v_lshlrev_b32_e32 v106, 16, v63
	v_and_b32_e32 v108, 0xffff0000, v63
	v_lshlrev_b32_e32 v111, 16, v44
	v_lshlrev_b32_e32 v110, 16, v32
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v44
	v_and_b32_e32 v112, 0xffff0000, v32
	v_pk_mul_f32 v[112:113], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v115, 16, v45
	v_lshlrev_b32_e32 v114, 16, v33
	v_pk_fma_f32 v[110:111], v[114:115], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v45
	v_and_b32_e32 v114, 0xffff0000, v33
	v_pk_fma_f32 v[112:113], v[114:115], v[100:101], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v46
	v_lshlrev_b32_e32 v114, 16, v34
	v_pk_fma_f32 v[110:111], v[114:115], v[102:103], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v46
	v_and_b32_e32 v114, 0xffff0000, v34
	v_pk_fma_f32 v[112:113], v[114:115], v[104:105], v[112:113] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v115, 16, v47
	v_lshlrev_b32_e32 v114, 16, v35
	v_pk_fma_f32 v[110:111], v[114:115], v[106:107], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v115, 0xffff0000, v47
	v_and_b32_e32 v114, 0xffff0000, v35
	v_pk_fma_f32 v[112:113], v[114:115], v[108:109], v[112:113] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[110:111], v[110:111], v[112:113]
	s_nop 0
	v_pk_add_f32 v[92:93], v[110:111], v[92:93]
	v_lshlrev_b32_e32 v111, 16, v20
	v_lshlrev_b32_e32 v110, 16, v12
	v_pk_mul_f32 v[110:111], v[88:89], v[110:111] op_sel_hi:[0,1]
	v_and_b32_e32 v113, 0xffff0000, v20
	v_and_b32_e32 v112, 0xffff0000, v12
	v_pk_mul_f32 v[96:97], v[96:97], v[112:113] op_sel_hi:[0,1]
	v_lshlrev_b32_e32 v113, 16, v21
	v_lshlrev_b32_e32 v112, 16, v13
	v_pk_fma_f32 v[98:99], v[112:113], v[98:99], v[110:111] op_sel_hi:[1,0,1]
	v_and_b32_e32 v111, 0xffff0000, v21
	v_and_b32_e32 v110, 0xffff0000, v13
	v_pk_fma_f32 v[96:97], v[110:111], v[100:101], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v22
	v_lshlrev_b32_e32 v100, 16, v14
	v_pk_fma_f32 v[98:99], v[100:101], v[102:103], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v22
	v_and_b32_e32 v100, 0xffff0000, v14
	v_pk_fma_f32 v[96:97], v[100:101], v[104:105], v[96:97] op_sel_hi:[1,0,1]
	v_lshlrev_b32_e32 v101, 16, v23
	v_lshlrev_b32_e32 v100, 16, v15
	v_pk_fma_f32 v[98:99], v[100:101], v[106:107], v[98:99] op_sel_hi:[1,0,1]
	v_and_b32_e32 v101, 0xffff0000, v23
	v_and_b32_e32 v100, 0xffff0000, v15
	v_pk_fma_f32 v[96:97], v[100:101], v[108:109], v[96:97] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[96:97], v[98:99], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[96:97], v[90:91]
	s_branch .LBB2_13
.LBB2_30:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mov_b32_e32 v93, v89
	v_mov_b32_e32 v92, v89
	v_mov_b32_e32 v91, v89
	v_mov_b32_e32 v90, v89
.LBB2_31:                               ;   in Loop: Header=BB2_11 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v93, v93, v93 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v92, v92, v92 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v91, v91, v91 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v90, v90, v90 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB2_10
; %bb.32:                               ;   in Loop: Header=BB2_11 Depth=1
	v_and_b32_e32 v83, 0x7f800000, v93
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.33:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v83, v93, 16, 1
	v_add3_u32 v88, v93, v83, s25
; %bb.34:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.35:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v93
	v_cmp_eq_u32_sdwa vcc, v93, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v83, v93, vcc
; %bb.36:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v83, v89
	v_lshl_add_u64 v[96:97], v[82:83], 1, s[8:9]
	global_store_short_d16_hi v[96:97], v88, off
	v_and_b32_e32 v83, 0x7f800000, v92
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.37:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v83, v92, 16, 1
	v_add3_u32 v83, v92, v83, s25
                                        ; implicit-def: $vgpr92
; %bb.38:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.39:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v92
	v_cmp_eq_u32_sdwa vcc, v92, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
; %bb.40:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v82
	v_lshl_add_u64 v[92:93], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[92:93], v83, off
	v_and_b32_e32 v83, 0x7f800000, v91
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.41:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v83, v91, 16, 1
	v_add3_u32 v83, v91, v83, s25
; %bb.42:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.43:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v91
	v_cmp_eq_u32_sdwa vcc, v91, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v91, vcc
; %bb.44:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[92:93], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[92:93], v83, off
	v_and_b32_e32 v83, 0x7f800000, v90
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.45:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v83, v90, 16, 1
	v_add3_u32 v83, v90, v83, s25
                                        ; implicit-def: $vgpr90
; %bb.46:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB2_9
; %bb.47:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v90
	v_cmp_eq_u32_sdwa vcc, v90, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v90, vcc
	s_branch .LBB2_9
.LBB2_48:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
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
		.amdhsa_next_free_vgpr 118
		.amdhsa_next_free_sgpr 27
		.amdhsa_accum_offset 120
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
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end2:
	.size	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end2-_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3516
; NumSgprs: 33
; NumVgprs: 118
; NumAgprs: 0
; TotalNumVgprs: 118
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 118
; AccumOffset: 120
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 29
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_2223a8cfc5787353,@object ; @__hip_cuid_2223a8cfc5787353
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2223a8cfc5787353
__hip_cuid_2223a8cfc5787353:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2223a8cfc5787353, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2223a8cfc5787353
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
      - .actual_access:  read_only
        .address_space:  global
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     101
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
      - .actual_access:  read_only
        .address_space:  global
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     118
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
