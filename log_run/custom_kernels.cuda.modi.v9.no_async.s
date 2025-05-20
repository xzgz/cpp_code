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
	.section	.text._Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,@function
_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii: ; @_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_mul_i32 s12, s12, s10
	v_bfe_u32 v2, v0, 10, 10
	v_add_u32_e32 v80, s12, v2
	v_cmp_gt_u32_e32 vcc, s3, v80
	v_add_u32_e32 v1, 1, v80
	v_cmp_le_u32_e64 s[0:1], s3, v1
	s_and_b64 s[12:13], vcc, s[0:1]
	v_mov_b32_e32 v83, 1
	s_and_saveexec_b64 s[0:1], s[12:13]
; %bb.1:
	v_subrev_u32_e32 v1, s3, v80
	v_cmp_eq_u32_e32 vcc, -1, v1
	s_nop 1
	v_cndmask_b32_e64 v83, 0, 1, vcc
	s_add_i32 s12, s3, -1
	v_mov_b32_e32 v80, s12
; %bb.2:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v3, 0x3ff, v0
	v_lshlrev_b32_e32 v82, 3, v3
	s_lshl_b32 s24, s2, 2
	s_mov_b32 s18, 0
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_eq_u32 s2, 0
	v_lshlrev_b32_e32 v100, 4, v3
	s_cbranch_scc1 .LBB2_8
; %bb.3:
	s_min_i32 s19, s24, 0x8000
	v_lshlrev_b32_e32 v0, 4, v3
	v_lshl_add_u32 v4, v2, 10, v0
	v_lshl_add_u32 v5, v2, 9, v82
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr14_sgpr15
	s_branch .LBB2_5
.LBB2_4:                                ;   in Loop: Header=BB2_5 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_b64 s[16:17], exec, s[14:15]
	s_or_b64 s[0:1], s[16:17], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_7
.LBB2_5:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s18, v5
	v_cmp_gt_u32_e32 vcc, s19, v0
	s_or_b64 s[14:15], s[14:15], exec
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_4
; %bb.6:                                ;   in Loop: Header=BB2_5 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s18, 0x2000
	s_cmp_ge_u32 s18, s19
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[14:15], s[14:15], s[20:21]
	s_branch .LBB2_4
.LBB2_7:
	s_or_b64 exec, exec, s[0:1]
.LBB2_8:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v2
	v_cmp_gt_u32_e64 s[0:1], s3, v80
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB2_97
; %bb.9:
	v_cmp_ne_u32_e32 vcc, 63, v3
	s_mul_i32 s25, s11, s10
	s_add_i32 s26, s3, -1
	s_sub_i32 s27, s25, s3
	s_add_i32 s27, s27, 2
	s_lshl_b32 s28, s2, 1
	s_mul_i32 s29, s2, 6
	v_add_u32_e32 v101, s28, v82
	v_mad_u64_u32 v[84:85], s[0:1], s2, 3, v[82:83]
	v_add_u32_e32 v85, s2, v82
	s_mov_b64 s[14:15], 0
	v_cndmask_b32_e64 v0, 0, 1, s[12:13]
	v_cmp_ne_u32_e64 s[0:1], 1, v0
	v_mov_b32_e32 v87, 0
	s_movk_i32 s30, 0x7fff
	s_xor_b64 s[16:17], vcc, -1
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $vgpr50_vgpr51
                                        ; implicit-def: $vgpr66_vgpr67
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $vgpr54_vgpr55
                                        ; implicit-def: $vgpr70_vgpr71
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $vgpr58_vgpr59
                                        ; implicit-def: $vgpr74_vgpr75
                                        ; implicit-def: $vgpr30_vgpr31
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $vgpr62_vgpr63
                                        ; implicit-def: $vgpr78_vgpr79
	s_branch .LBB2_11
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v81, s25, v80
	v_cmp_le_u32_e32 vcc, s3, v81
	v_add_u32_e32 v86, 1, v81
	v_cmp_gt_u32_e64 s[10:11], s3, v86
	v_add_u32_e32 v80, s27, v80
	v_cmp_eq_u32_e64 s[12:13], 1, v80
	v_mov_b32_e32 v80, s26
	s_or_b64 vcc, vcc, s[10:11]
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_cmp_le_u32_e64 s[10:11], s3, v80
	s_or_b64 vcc, vcc, s[12:13]
	s_or_b64 s[14:15], s[10:11], s[14:15]
	v_cndmask_b32_e32 v83, 0, v83, vcc
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB2_97
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_17 Depth 2
	s_and_b64 vcc, exec, s[0:1]
	s_mov_b32 s31, 0
	s_cbranch_vccnz .LBB2_94
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mad_u64_u32 v[88:89], s[10:11], v80, s2, v[82:83]
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v81, 0
	s_branch .LBB2_17
.LBB2_13:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_14:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_15:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[12:13]
.LBB2_16:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s31, 0x800
	s_cmp_ge_u32 s31, s2
	v_add_u32_e32 v104, 0x1000, v104
	s_cbranch_scc1 .LBB2_95
.LBB2_17:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v90, s31, v82
	v_cmp_gt_u32_e32 vcc, s2, v90
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB2_89
; %bb.18:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v86, s31, v88
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[86:87], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	v_cmp_lt_u32_e64 s[10:11], s30, v90
                                        ; implicit-def: $vgpr16_vgpr17
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_xor_b64 s[10:11], exec, s[18:19]
	s_cbranch_execz .LBB2_20
; %bb.19:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v91, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[16:17], v[90:91], 1, s[6:7]
	global_load_dwordx4 v[16:19], v[16:17], off
.LBB2_20:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_22
; %bb.21:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_read_b128 v[16:19], v104
.LBB2_22:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v96, s31, v85
	v_cmp_lt_u32_e64 s[10:11], s30, v96
                                        ; implicit-def: $vgpr32_vgpr33
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_xor_b64 s[10:11], exec, s[18:19]
	s_cbranch_execz .LBB2_24
; %bb.23:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v97, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[32:33], v[96:97], 1, s[6:7]
	global_load_dwordx4 v[32:35], v[32:33], off
.LBB2_24:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_26
; %bb.25:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e32 v32, s28, v104
	ds_read_b128 v[32:35], v32
.LBB2_26:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v94, s31, v101
	v_cmp_lt_u32_e64 s[10:11], s30, v94
                                        ; implicit-def: $vgpr48_vgpr49
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_xor_b64 s[10:11], exec, s[18:19]
	s_cbranch_execz .LBB2_28
; %bb.27:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v95, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[48:49], v[94:95], 1, s[6:7]
	global_load_dwordx4 v[48:51], v[48:49], off
.LBB2_28:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_30
; %bb.29:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e32 v48, s24, v104
	ds_read_b128 v[48:51], v48
.LBB2_30:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v92, s31, v84
	v_cmp_lt_u32_e64 s[10:11], s30, v92
                                        ; implicit-def: $vgpr64_vgpr65
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_xor_b64 s[10:11], exec, s[18:19]
	s_cbranch_execz .LBB2_32
; %bb.31:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v93, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[64:65], v[92:93], 1, s[6:7]
	global_load_dwordx4 v[64:67], v[64:65], off
.LBB2_32:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_34
; %bb.33:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e32 v64, s29, v104
	ds_read_b128 v[64:67], v64
.LBB2_34:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v98, 0x200, v90
	v_cmp_gt_u32_e64 s[10:11], s2, v98
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_88
; %bb.35:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v8, 0x200, v86
	v_mov_b32_e32 v9, v87
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	v_cmp_lt_u32_e64 s[10:11], s30, v98
                                        ; implicit-def: $vgpr20_vgpr21
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_xor_b64 s[10:11], exec, s[20:21]
	s_cbranch_execz .LBB2_37
; %bb.36:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v99, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[20:21], v[98:99], 1, s[6:7]
	global_load_dwordx4 v[20:23], v[20:21], off
.LBB2_37:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_39
; %bb.38:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_read_b128 v[20:23], v104 offset:1024
.LBB2_39:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v38, 0x200, v96
	v_cmp_lt_u32_e64 s[10:11], s30, v38
                                        ; implicit-def: $vgpr36_vgpr37
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_xor_b64 s[10:11], exec, s[20:21]
	s_cbranch_execz .LBB2_41
; %bb.40:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v39, v87
	v_lshl_add_u64 v[36:37], v[38:39], 1, s[6:7]
	global_load_dwordx4 v[36:39], v[36:37], off
.LBB2_41:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_43
; %bb.42:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v36, s28, v104
	ds_read_b128 v[36:39], v36 offset:1024
.LBB2_43:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v54, 0x200, v94
	v_cmp_lt_u32_e64 s[10:11], s30, v54
                                        ; implicit-def: $vgpr52_vgpr53
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_xor_b64 s[10:11], exec, s[20:21]
	s_cbranch_execz .LBB2_45
; %bb.44:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v55, v87
	v_lshl_add_u64 v[52:53], v[54:55], 1, s[6:7]
	global_load_dwordx4 v[52:55], v[52:53], off
.LBB2_45:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_47
; %bb.46:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v52, s24, v104
	ds_read_b128 v[52:55], v52 offset:1024
.LBB2_47:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v70, 0x200, v92
	v_cmp_lt_u32_e64 s[10:11], s30, v70
                                        ; implicit-def: $vgpr68_vgpr69
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_xor_b64 s[10:11], exec, s[20:21]
	s_cbranch_execz .LBB2_49
; %bb.48:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v71, v87
	v_lshl_add_u64 v[68:69], v[70:71], 1, s[6:7]
	global_load_dwordx4 v[68:71], v[68:69], off
.LBB2_49:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_51
; %bb.50:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v68, s29, v104
	ds_read_b128 v[68:71], v68 offset:1024
.LBB2_51:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v98, 0x400, v90
	v_cmp_gt_u32_e64 s[10:11], s2, v98
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB2_87
; %bb.52:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v4, 0x400, v86
	v_mov_b32_e32 v5, v87
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	v_cmp_lt_u32_e64 s[10:11], s30, v98
                                        ; implicit-def: $vgpr24_vgpr25
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_xor_b64 s[10:11], exec, s[22:23]
	s_cbranch_execz .LBB2_54
; %bb.53:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v99, v87
	v_lshl_add_u64 v[24:25], v[98:99], 1, s[6:7]
	global_load_dwordx4 v[24:27], v[24:25], off
.LBB2_54:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_56
; %bb.55:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[24:27], v104 offset:2048
.LBB2_56:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v42, 0x400, v96
	v_cmp_lt_u32_e64 s[10:11], s30, v42
                                        ; implicit-def: $vgpr40_vgpr41
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_xor_b64 s[10:11], exec, s[22:23]
	s_cbranch_execz .LBB2_58
; %bb.57:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v43, v87
	v_lshl_add_u64 v[40:41], v[42:43], 1, s[6:7]
	global_load_dwordx4 v[40:43], v[40:41], off
.LBB2_58:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_60
; %bb.59:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v40, s28, v104
	ds_read_b128 v[40:43], v40 offset:2048
.LBB2_60:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v58, 0x400, v94
	v_cmp_lt_u32_e64 s[10:11], s30, v58
                                        ; implicit-def: $vgpr56_vgpr57
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_xor_b64 s[10:11], exec, s[22:23]
	s_cbranch_execz .LBB2_62
; %bb.61:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v59, v87
	v_lshl_add_u64 v[56:57], v[58:59], 1, s[6:7]
	global_load_dwordx4 v[56:59], v[56:57], off
.LBB2_62:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_64
; %bb.63:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v56, s24, v104
	ds_read_b128 v[56:59], v56 offset:2048
.LBB2_64:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v74, 0x400, v92
	v_cmp_lt_u32_e64 s[10:11], s30, v74
                                        ; implicit-def: $vgpr72_vgpr73
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_xor_b64 s[10:11], exec, s[22:23]
	s_cbranch_execz .LBB2_66
; %bb.65:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v75, v87
	v_lshl_add_u64 v[72:73], v[74:75], 1, s[6:7]
	global_load_dwordx4 v[72:75], v[72:73], off
.LBB2_66:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_68
; %bb.67:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v72, s29, v104
	ds_read_b128 v[72:75], v72 offset:2048
.LBB2_68:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v98, 0x600, v90
	v_cmp_gt_u32_e64 s[10:11], s2, v98
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_cbranch_execz .LBB2_86
; %bb.69:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v86, 0x600, v86
	v_lshl_add_u64 v[0:1], v[86:87], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	v_cmp_lt_u32_e64 s[10:11], s30, v98
                                        ; implicit-def: $vgpr28_vgpr29
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_xor_b64 s[10:11], exec, s[34:35]
	s_cbranch_execz .LBB2_71
; %bb.70:                               ;   in Loop: Header=BB2_17 Depth=2
	v_mov_b32_e32 v99, v87
	v_lshl_add_u64 v[28:29], v[98:99], 1, s[6:7]
	global_load_dwordx4 v[28:31], v[28:29], off
.LBB2_71:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_73
; %bb.72:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	ds_read_b128 v[28:31], v104 offset:3072
.LBB2_73:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v86, 0x600, v96
	v_cmp_lt_u32_e64 s[10:11], s30, v86
                                        ; implicit-def: $vgpr44_vgpr45
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_xor_b64 s[10:11], exec, s[34:35]
	s_cbranch_execz .LBB2_75
; %bb.74:                               ;   in Loop: Header=BB2_17 Depth=2
	v_lshl_add_u64 v[44:45], v[86:87], 1, s[6:7]
	global_load_dwordx4 v[44:47], v[44:45], off
.LBB2_75:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_77
; %bb.76:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v44, s28, v104
	ds_read_b128 v[44:47], v44 offset:3072
.LBB2_77:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v86, 0x600, v94
	v_cmp_lt_u32_e64 s[10:11], s30, v86
                                        ; implicit-def: $vgpr60_vgpr61
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_xor_b64 s[10:11], exec, s[34:35]
	s_cbranch_execz .LBB2_79
; %bb.78:                               ;   in Loop: Header=BB2_17 Depth=2
	v_lshl_add_u64 v[60:61], v[86:87], 1, s[6:7]
	global_load_dwordx4 v[60:63], v[60:61], off
.LBB2_79:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_81
; %bb.80:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v60, s24, v104
	ds_read_b128 v[60:63], v60 offset:3072
.LBB2_81:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v86, 0x600, v92
	v_cmp_lt_u32_e64 s[10:11], s30, v86
                                        ; implicit-def: $vgpr76_vgpr77
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_xor_b64 s[10:11], exec, s[34:35]
	s_cbranch_execz .LBB2_83
; %bb.82:                               ;   in Loop: Header=BB2_17 Depth=2
	v_lshl_add_u64 v[76:77], v[86:87], 1, s[6:7]
	global_load_dwordx4 v[76:79], v[76:77], off
.LBB2_83:                               ;   in Loop: Header=BB2_17 Depth=2
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_85
; %bb.84:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v76, s29, v104
	ds_read_b128 v[76:79], v76 offset:3072
.LBB2_85:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
.LBB2_86:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB2_87:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_88:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_89:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_16
; %bb.90:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v16, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v17, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v18, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v19, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v32, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v33, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v34, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v35, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v48, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v49, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v50, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v51, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v64, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v65, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v66, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v67, v15
	;;#ASMEND
	v_add_u32_e32 v86, 0x200, v90
	v_cmp_gt_u32_e32 vcc, s2, v86
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB2_15
; %bb.91:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v20, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v21, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v22, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v36, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v37, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v38, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v39, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v52, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v53, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v54, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v55, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v68, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v69, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v70, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v71, v11
	;;#ASMEND
	v_add_u32_e32 v86, 0x400, v90
	v_cmp_gt_u32_e32 vcc, s2, v86
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_14
; %bb.92:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v24, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v25, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v26, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v27, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v40, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v41, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v42, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v43, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v56, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v57, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v58, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v59, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v72, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v73, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v74, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v75, v7
	;;#ASMEND
	v_add_u32_e32 v86, 0x600, v90
	v_cmp_gt_u32_e32 vcc, s2, v86
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB2_13
; %bb.93:                               ;   in Loop: Header=BB2_17 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v28, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v29, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v30, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v81, v31, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v44, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v45, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v46, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v103, v47, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v60, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v61, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v62, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v102, v63, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v76, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v77, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v78, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v79, v3
	;;#ASMEND
	s_branch .LBB2_13
.LBB2_94:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mov_b32_e32 v81, v87
	v_mov_b32_e32 v103, v87
	v_mov_b32_e32 v102, v87
	v_mov_b32_e32 v89, v87
.LBB2_95:                               ;   in Loop: Header=BB2_11 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v103, v103, v103 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v102, v102, v102 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v89, v89, v89 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_and_b64 s[12:13], s[16:17], vcc
	s_and_saveexec_b64 s[10:11], s[12:13]
	s_cbranch_execz .LBB2_10
; %bb.96:                               ;   in Loop: Header=BB2_11 Depth=1
	v_cvt_f16_f32_e32 v86, v81
	v_mov_b32_e32 v81, v87
	v_cvt_f16_f32_e32 v88, v103
	v_lshl_add_u64 v[90:91], v[80:81], 1, s[8:9]
	global_store_short v[90:91], v86, off
	v_add_u32_e32 v86, s3, v80
	v_lshl_add_u64 v[90:91], v[86:87], 1, s[8:9]
	global_store_short v[90:91], v88, off
	v_cvt_f16_f32_e32 v81, v102
	v_add_u32_e32 v86, s3, v86
	v_lshl_add_u64 v[90:91], v[86:87], 1, s[8:9]
	v_cvt_f16_f32_e32 v92, v89
	global_store_short v[90:91], v81, off
	v_add_u32_e32 v86, s3, v86
	v_lshl_add_u64 v[88:89], v[86:87], 1, s[8:9]
	global_store_short v[88:89], v92, off
	s_branch .LBB2_10
.LBB2_97:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
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
		.amdhsa_next_free_vgpr 105
		.amdhsa_next_free_sgpr 36
		.amdhsa_accum_offset 108
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
	.section	.text._Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end2:
	.size	_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end2-_Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3288
; NumSgprs: 42
; NumVgprs: 105
; NumAgprs: 0
; TotalNumVgprs: 105
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 105
; AccumOffset: 108
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 26
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
	s_cbranch_scc1 .LBB3_6
; %bb.1:
	s_lshl_b32 s0, s2, 2
	s_min_i32 s20, s0, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v80
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB3_3
.LBB3_2:                                ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB3_5
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s20, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB3_2
; %bb.4:                                ;   in Loop: Header=BB3_3 Depth=1
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
	s_branch .LBB3_2
.LBB3_5:
	s_or_b64 exec, exec, s[0:1]
.LBB3_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_48
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v82, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v82
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB3_48
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
	s_branch .LBB3_11
.LBB3_9:                                ;   in Loop: Header=BB3_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[90:91], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[90:91], v83, off
.LBB3_10:                               ;   in Loop: Header=BB3_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v82, s22, v82
	v_cmp_le_u32_e32 vcc, s3, v82
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v84, s23, v84
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB3_48
.LBB3_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_17 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB3_30
; %bb.12:                               ;   in Loop: Header=BB3_11 Depth=1
	s_mov_b32 s26, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, v90
	v_mov_b32_e32 v92, v90
	v_mov_b32_e32 v93, v90
	s_branch .LBB3_17
.LBB3_13:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB3_14:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB3_15:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s26, 0x800
	s_cmp_ge_u32 s26, s2
	s_cbranch_scc1 .LBB3_31
.LBB3_17:                               ;   Parent Loop BB3_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v83, s26, v80
	v_cmp_gt_u32_e32 vcc, s2, v83
	v_add_u32_e32 v96, 0x200, v83
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB3_25
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_24
; %bb.19:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_23
; %bb.20:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_22
; %bb.21:                               ;   in Loop: Header=BB3_17 Depth=2
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
.LBB3_22:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB3_23:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB3_24:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB3_25:                               ;   in Loop: Header=BB3_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB3_16
; %bb.26:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_15
; %bb.27:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_14
; %bb.28:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_cbranch_execz .LBB3_13
; %bb.29:                               ;   in Loop: Header=BB3_17 Depth=2
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
	s_branch .LBB3_13
.LBB3_30:                               ;   in Loop: Header=BB3_11 Depth=1
	v_mov_b32_e32 v93, v89
	v_mov_b32_e32 v92, v89
	v_mov_b32_e32 v91, v89
	v_mov_b32_e32 v90, v89
.LBB3_31:                               ;   in Loop: Header=BB3_11 Depth=1
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
	s_cbranch_execz .LBB3_10
; %bb.32:                               ;   in Loop: Header=BB3_11 Depth=1
	v_and_b32_e32 v83, 0x7f800000, v93
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.33:                               ;   in Loop: Header=BB3_11 Depth=1
	v_bfe_u32 v83, v93, 16, 1
	v_add3_u32 v88, v93, v83, s25
; %bb.34:                               ;   in Loop: Header=BB3_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.35:                               ;   in Loop: Header=BB3_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v93
	v_cmp_eq_u32_sdwa vcc, v93, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v83, v93, vcc
; %bb.36:                               ;   in Loop: Header=BB3_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v83, v89
	v_lshl_add_u64 v[96:97], v[82:83], 1, s[8:9]
	global_store_short_d16_hi v[96:97], v88, off
	v_and_b32_e32 v83, 0x7f800000, v92
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.37:                               ;   in Loop: Header=BB3_11 Depth=1
	v_bfe_u32 v83, v92, 16, 1
	v_add3_u32 v83, v92, v83, s25
                                        ; implicit-def: $vgpr92
; %bb.38:                               ;   in Loop: Header=BB3_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.39:                               ;   in Loop: Header=BB3_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v92
	v_cmp_eq_u32_sdwa vcc, v92, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
; %bb.40:                               ;   in Loop: Header=BB3_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v82
	v_lshl_add_u64 v[92:93], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[92:93], v83, off
	v_and_b32_e32 v83, 0x7f800000, v91
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.41:                               ;   in Loop: Header=BB3_11 Depth=1
	v_bfe_u32 v83, v91, 16, 1
	v_add3_u32 v83, v91, v83, s25
; %bb.42:                               ;   in Loop: Header=BB3_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.43:                               ;   in Loop: Header=BB3_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v91
	v_cmp_eq_u32_sdwa vcc, v91, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v91, vcc
; %bb.44:                               ;   in Loop: Header=BB3_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v88, s3, v88
	v_lshl_add_u64 v[92:93], v[88:89], 1, s[8:9]
	global_store_short_d16_hi v[92:93], v83, off
	v_and_b32_e32 v83, 0x7f800000, v90
	v_cmp_ne_u32_e32 vcc, s24, v83
                                        ; implicit-def: $vgpr83
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.45:                               ;   in Loop: Header=BB3_11 Depth=1
	v_bfe_u32 v83, v90, 16, 1
	v_add3_u32 v83, v90, v83, s25
                                        ; implicit-def: $vgpr90
; %bb.46:                               ;   in Loop: Header=BB3_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB3_9
; %bb.47:                               ;   in Loop: Header=BB3_11 Depth=1
	v_or_b32_e32 v83, 0x10000, v90
	v_cmp_eq_u32_sdwa vcc, v90, v89 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v90, vcc
	s_branch .LBB3_9
.LBB3_48:
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
.Lfunc_end3:
	.size	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end3-_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
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
	.section	.text._Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,@function
_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii: ; @_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_mul_i32 s12, s12, s10
	v_bfe_u32 v2, v0, 10, 10
	v_add_u32_e32 v72, s12, v2
	v_cmp_gt_u32_e32 vcc, s3, v72
	v_add_u32_e32 v1, 1, v72
	v_cmp_le_u32_e64 s[0:1], s3, v1
	s_and_b64 s[12:13], vcc, s[0:1]
	v_mov_b32_e32 v75, 1
	s_and_saveexec_b64 s[0:1], s[12:13]
; %bb.1:
	v_subrev_u32_e32 v1, s3, v72
	v_cmp_eq_u32_e32 vcc, -1, v1
	s_nop 1
	v_cndmask_b32_e64 v75, 0, 1, vcc
	s_add_i32 s12, s3, -1
	v_mov_b32_e32 v72, s12
; %bb.2:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v3, 0x3ff, v0
	v_lshlrev_b32_e32 v74, 3, v3
	s_lshl_b32 s24, s2, 2
	s_mov_b32 s18, 0
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_eq_u32 s2, 0
	v_lshlrev_b32_e32 v88, 4, v3
	s_cbranch_scc1 .LBB4_8
; %bb.3:
	s_min_i32 s19, s24, 0x8000
	v_lshlrev_b32_e32 v0, 4, v3
	v_lshl_add_u32 v4, v2, 10, v0
	v_lshl_add_u32 v5, v2, 9, v74
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr14_sgpr15
	s_branch .LBB4_5
.LBB4_4:                                ;   in Loop: Header=BB4_5 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_b64 s[16:17], exec, s[14:15]
	s_or_b64 s[0:1], s[16:17], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB4_7
.LBB4_5:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s18, v5
	v_cmp_gt_u32_e32 vcc, s19, v0
	s_or_b64 s[14:15], s[14:15], exec
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB4_4
; %bb.6:                                ;   in Loop: Header=BB4_5 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s18, 0x2000
	s_cmp_ge_u32 s18, s19
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[14:15], s[14:15], s[20:21]
	s_branch .LBB4_4
.LBB4_7:
	s_or_b64 exec, exec, s[0:1]
.LBB4_8:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v2
	v_cmp_gt_u32_e64 s[0:1], s3, v72
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB4_115
; %bb.9:
	v_cmp_eq_u32_e64 s[0:1], 63, v3
	s_mul_i32 s25, s11, s10
	s_add_i32 s26, s3, -1
	s_sub_i32 s27, s25, s3
	s_add_i32 s27, s27, 2
	s_lshl_b32 s28, s2, 1
	s_mul_i32 s29, s2, 6
	v_add_u32_e32 v89, s28, v74
	v_mad_u64_u32 v[76:77], s[10:11], s2, 3, v[74:75]
	v_add_u32_e32 v77, s2, v74
	s_mov_b64 s[16:17], 0
	v_cndmask_b32_e64 v0, 0, 1, s[12:13]
	v_cmp_ne_u32_e64 s[10:11], 1, v0
	v_mov_b32_e32 v79, 0
	s_movk_i32 s30, 0x7fff
	s_mov_b32 s31, 0xffff
	s_mov_b32 s33, 0x5040100
	s_mov_b32 s34, 0x7f800000
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr46
                                        ; implicit-def: $vgpr43
                                        ; implicit-def: $vgpr42
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr34
                                        ; implicit-def: $vgpr90
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr109
                                        ; implicit-def: $vgpr110
                                        ; implicit-def: $vgpr114
                                        ; implicit-def: $vgpr115
                                        ; implicit-def: $vgpr47
                                        ; implicit-def: $vgpr48
                                        ; implicit-def: $vgpr96
                                        ; implicit-def: $vgpr97
                                        ; implicit-def: $vgpr98
                                        ; implicit-def: $vgpr99
                                        ; implicit-def: $vgpr49
                                        ; implicit-def: $vgpr41
                                        ; implicit-def: $vgpr91
                                        ; implicit-def: $vgpr87
                                        ; implicit-def: $vgpr28
                                        ; implicit-def: $vgpr94
                                        ; implicit-def: $vgpr93
                                        ; implicit-def: $vgpr52
                                        ; implicit-def: $vgpr92
                                        ; implicit-def: $vgpr31
                                        ; implicit-def: $vgpr95
                                        ; implicit-def: $vgpr55
                                        ; implicit-def: $vgpr38
                                        ; implicit-def: $vgpr58
                                        ; implicit-def: $vgpr29
                                        ; implicit-def: $vgpr37
                                        ; implicit-def: $vgpr32
                                        ; implicit-def: $vgpr44
                                        ; implicit-def: $vgpr39
                                        ; implicit-def: $vgpr59
                                        ; implicit-def: $vgpr105
                                        ; implicit-def: $vgpr101
                                        ; implicit-def: $vgpr53
                                        ; implicit-def: $vgpr100
                                        ; implicit-def: $vgpr56
                                        ; implicit-def: $vgpr45
                                        ; implicit-def: $vgpr108
                                        ; implicit-def: $vgpr107
                                        ; implicit-def: $vgpr117
                                        ; implicit-def: $vgpr116
                                        ; implicit-def: $vgpr33
                                        ; implicit-def: $vgpr57
                                        ; implicit-def: $vgpr40
                                        ; implicit-def: $vgpr112
                                        ; implicit-def: $vgpr106
                                        ; implicit-def: $vgpr102
                                        ; implicit-def: $vgpr104
                                        ; implicit-def: $vgpr103
                                        ; implicit-def: $vgpr113
                                        ; implicit-def: $vgpr111
                                        ; implicit-def: $vgpr119
                                        ; implicit-def: $vgpr118
	s_branch .LBB4_12
.LBB4_10:                               ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v78, s3, v78
	v_lshl_add_u64 v[64:65], v[78:79], 1, s[8:9]
	global_store_short_d16_hi v[64:65], v63, off
.LBB4_11:                               ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v63, s25, v72
	v_cmp_le_u32_e32 vcc, s3, v63
	v_add_u32_e32 v64, 1, v63
	v_cmp_gt_u32_e64 s[12:13], s3, v64
	v_add_u32_e32 v64, s27, v72
	v_cmp_eq_u32_e64 s[14:15], 1, v64
	v_mov_b32_e32 v64, s26
	s_or_b64 vcc, vcc, s[12:13]
	v_cndmask_b32_e32 v72, v64, v63, vcc
	s_or_b64 vcc, vcc, s[14:15]
	v_cndmask_b32_e32 v75, 0, v75, vcc
	v_cmp_le_u32_e32 vcc, s3, v72
	v_perm_b32 v110, v62, v110, s33
	v_perm_b32 v115, v61, v115, s33
	v_perm_b32 v47, v60, v47, s33
	v_perm_b32 v48, v36, v48, s33
	v_perm_b32 v97, v23, v97, s33
	v_perm_b32 v99, v22, v99, s33
	v_perm_b32 v49, v19, v49, s33
	s_or_b64 s[16:17], vcc, s[16:17]
	v_perm_b32 v41, v18, v41, s33
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execz .LBB4_115
.LBB4_12:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_18 Depth 2
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB4_96
; %bb.13:                               ;   in Loop: Header=BB4_12 Depth=1
	v_mad_u64_u32 v[84:85], s[12:13], v72, s2, v[74:75]
	s_mov_b32 s35, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v73, v88
	v_mov_b32_e32 v81, v80
	v_mov_b32_e32 v82, v80
	v_mov_b32_e32 v83, v80
	s_branch .LBB4_18
.LBB4_14:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB4_15:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB4_16:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB4_17:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s35, 0x800
	s_cmp_ge_u32 s35, s2
	v_add_u32_e32 v73, 0x1000, v73
	s_cbranch_scc1 .LBB4_95
.LBB4_18:                               ;   Parent Loop BB4_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v86, s35, v74
	v_cmp_gt_u32_e32 vcc, s2, v86
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_90
; %bb.19:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v78, s35, v84
	v_lshl_add_u64 v[12:13], v[78:79], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	v_cmp_lt_u32_e64 s[12:13], s30, v86
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr87
                                        ; implicit-def: $vgpr91
                                        ; implicit-def: $vgpr24
                                        ; implicit-def: $vgpr19
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_xor_b64 s[12:13], exec, s[18:19]
	s_cbranch_execz .LBB4_21
; %bb.20:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v87, v79
	v_lshl_add_u64 v[18:19], v[86:87], 1, s[6:7]
	global_load_dwordx4 v[24:27], v[18:19], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v87, v25, v24, 16
	v_alignbit_b32 v91, v26, v25, 16
	v_alignbit_b32 v25, v27, v26, 16
	v_lshrrev_b32_e32 v19, 16, v27
.LBB4_21:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_23
; %bb.22:                               ;   in Loop: Header=BB4_18 Depth=2
	ds_read_b128 v[24:27], v73
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v87, v25, v24, 16
	v_alignbit_b32 v91, v26, v25, 16
	v_alignbit_b32 v25, v27, v26, 16
	v_lshrrev_b32_e32 v19, 16, v27
.LBB4_23:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v68, s35, v77
	v_cmp_lt_u32_e64 s[12:13], s30, v68
                                        ; implicit-def: $vgpr29
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_xor_b64 s[12:13], exec, s[18:19]
	s_cbranch_execz .LBB4_25
; %bb.24:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v69, v79
	v_lshl_add_u64 v[22:23], v[68:69], 1, s[6:7]
	global_load_dwordx4 v[26:29], v[22:23], off
.LBB4_25:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_27
; %bb.26:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s28, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[26:29], v18
.LBB4_27:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v62, s35, v89
	v_cmp_lt_u32_e64 s[12:13], s30, v62
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr93
                                        ; implicit-def: $vgpr94
                                        ; implicit-def: $vgpr34
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_xor_b64 s[12:13], exec, s[18:19]
	s_cbranch_execz .LBB4_29
; %bb.28:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v63, v79
	v_lshl_add_u64 v[22:23], v[62:63], 1, s[6:7]
	global_load_dwordx4 v[34:37], v[22:23], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v93, v35, v34, 16
	v_alignbit_b32 v94, v36, v35, 16
	v_alignbit_b32 v35, v37, v36, 16
	v_lshrrev_b32_e32 v22, 16, v37
.LBB4_29:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_31
; %bb.30:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s24, v73
	ds_read_b128 v[34:37], v18
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v93, v35, v34, 16
	v_alignbit_b32 v94, v36, v35, 16
	v_alignbit_b32 v35, v37, v36, 16
	v_lshrrev_b32_e32 v22, 16, v37
.LBB4_31:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v36, s35, v76
	v_cmp_lt_u32_e64 s[12:13], s30, v36
                                        ; implicit-def: $vgpr53
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_xor_b64 s[12:13], exec, s[18:19]
	s_cbranch_execz .LBB4_33
; %bb.32:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v37, v79
	v_lshl_add_u64 v[50:51], v[36:37], 1, s[6:7]
	global_load_dwordx4 v[50:53], v[50:51], off
.LBB4_33:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_35
; %bb.34:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s29, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[50:53], v18
.LBB4_35:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v18, 0x200, v86
	v_cmp_gt_u32_e64 s[12:13], s2, v18
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_bfi_b32 v100, s31, v53, v35
	v_alignbit_b32 v53, v22, v53, 16
	v_bfi_b32 v37, s31, v29, v25
	v_alignbit_b32 v29, v19, v29, 16
	s_and_saveexec_b64 s[18:19], s[12:13]
	s_cbranch_execz .LBB4_89
; %bb.36:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v8, 0x200, v78
	v_mov_b32_e32 v9, v79
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	v_cmp_lt_u32_e64 s[12:13], s30, v18
                                        ; implicit-def: $vgpr21
                                        ; implicit-def: $vgpr92
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr60
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_xor_b64 s[12:13], exec, s[20:21]
	s_cbranch_execz .LBB4_38
; %bb.37:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v19, v79
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	global_load_dwordx4 v[20:23], v[18:19], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v92, v21, v20, 16
	v_alignbit_b32 v21, v22, v21, 16
	v_lshrrev_b32_e32 v22, 16, v22
	v_lshrrev_b32_e32 v60, 16, v23
.LBB4_38:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_40
; %bb.39:                               ;   in Loop: Header=BB4_18 Depth=2
	ds_read_b128 v[20:23], v73 offset:1024
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v92, v21, v20, 16
	v_alignbit_b32 v21, v22, v21, 16
	v_lshrrev_b32_e32 v22, 16, v22
	v_lshrrev_b32_e32 v60, 16, v23
.LBB4_40:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v18, 0x200, v68
	v_cmp_lt_u32_e64 s[12:13], s30, v18
                                        ; implicit-def: $vgpr32
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_xor_b64 s[12:13], exec, s[20:21]
	s_cbranch_execz .LBB4_42
; %bb.41:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v19, v79
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	global_load_dwordx4 v[30:33], v[18:19], off
.LBB4_42:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_44
; %bb.43:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s28, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[30:33], v18 offset:1024
.LBB4_44:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v18, 0x200, v62
	v_cmp_lt_u32_e64 s[12:13], s30, v18
                                        ; implicit-def: $vgpr43
                                        ; implicit-def: $vgpr95
                                        ; implicit-def: $vgpr42
                                        ; implicit-def: $vgpr44
                                        ; implicit-def: $vgpr61
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_xor_b64 s[12:13], exec, s[20:21]
	s_cbranch_execz .LBB4_46
; %bb.45:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v19, v79
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	global_load_dwordx4 v[42:45], v[18:19], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v95, v43, v42, 16
	v_alignbit_b32 v43, v44, v43, 16
	v_lshrrev_b32_e32 v44, 16, v44
	v_lshrrev_b32_e32 v61, 16, v45
.LBB4_46:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_48
; %bb.47:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s24, v73
	ds_read_b128 v[42:45], v18 offset:1024
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v95, v43, v42, 16
	v_alignbit_b32 v43, v44, v43, 16
	v_lshrrev_b32_e32 v44, 16, v44
	v_lshrrev_b32_e32 v61, 16, v45
.LBB4_48:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v18, 0x200, v36
	v_cmp_lt_u32_e64 s[12:13], s30, v18
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_xor_b64 s[12:13], exec, s[20:21]
	s_cbranch_execz .LBB4_50
; %bb.49:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v19, v79
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	global_load_dwordx4 v[54:57], v[18:19], off
.LBB4_50:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_52
; %bb.51:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s29, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[54:57], v18 offset:1024
.LBB4_52:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v18, 0x400, v86
	v_cmp_gt_u32_e64 s[12:13], s2, v18
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_perm_b32 v103, v45, v57, s33
	v_bfi_b32 v45, s31, v56, v43
	v_alignbit_b32 v104, v61, v57, 16
	v_alignbit_b32 v56, v44, v56, 16
	v_perm_b32 v57, v23, v33, s33
	v_bfi_b32 v44, s31, v32, v21
	v_alignbit_b32 v33, v60, v33, 16
	v_alignbit_b32 v32, v22, v32, 16
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_cbranch_execz .LBB4_88
; %bb.53:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v4, 0x400, v78
	v_mov_b32_e32 v5, v79
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	v_cmp_lt_u32_e64 s[12:13], s30, v18
                                        ; implicit-def: $vgpr90
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr64
                                        ; implicit-def: $vgpr65
	s_and_saveexec_b64 s[22:23], s[12:13]
	s_xor_b64 s[12:13], exec, s[22:23]
	s_cbranch_execz .LBB4_55
; %bb.54:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v19, v79
	v_lshl_add_u64 v[16:17], v[18:19], 1, s[6:7]
	global_load_dwordx4 v[16:19], v[16:17], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v90, v17, v16, 16
	v_lshrrev_b32_e32 v63, 16, v17
	v_lshrrev_b32_e32 v64, 16, v18
	v_lshrrev_b32_e32 v65, 16, v19
.LBB4_55:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_57
; %bb.56:                               ;   in Loop: Header=BB4_18 Depth=2
	ds_read_b128 v[16:19], v73 offset:2048
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v90, v17, v16, 16
	v_lshrrev_b32_e32 v63, 16, v17
	v_lshrrev_b32_e32 v64, 16, v18
	v_lshrrev_b32_e32 v65, 16, v19
.LBB4_57:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v22, 0x400, v68
	v_cmp_lt_u32_e64 s[12:13], s30, v22
                                        ; implicit-def: $vgpr39
	s_and_saveexec_b64 s[22:23], s[12:13]
	s_xor_b64 s[12:13], exec, s[22:23]
	s_cbranch_execz .LBB4_59
; %bb.58:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v23, v79
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	global_load_dwordx4 v[38:41], v[22:23], off
.LBB4_59:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_61
; %bb.60:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v17, s28, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[38:41], v17 offset:2048
.LBB4_61:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v22, 0x400, v62
	v_cmp_lt_u32_e64 s[12:13], s30, v22
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr46
                                        ; implicit-def: $vgpr66
                                        ; implicit-def: $vgpr67
                                        ; implicit-def: $vgpr69
	s_and_saveexec_b64 s[22:23], s[12:13]
	s_xor_b64 s[12:13], exec, s[22:23]
	s_cbranch_execz .LBB4_63
; %bb.62:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v23, v79
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	global_load_dwordx4 v[46:49], v[22:23], off
	s_waitcnt vmcnt(0)
	v_alignbit_b32 v17, v47, v46, 16
	v_lshrrev_b32_e32 v66, 16, v47
	v_lshrrev_b32_e32 v67, 16, v48
	v_lshrrev_b32_e32 v69, 16, v49
.LBB4_63:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_65
; %bb.64:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v17, s24, v73
	ds_read_b128 v[46:49], v17 offset:2048
	s_waitcnt lgkmcnt(0)
	v_alignbit_b32 v17, v47, v46, 16
	v_lshrrev_b32_e32 v66, 16, v47
	v_lshrrev_b32_e32 v67, 16, v48
	v_lshrrev_b32_e32 v69, 16, v49
.LBB4_65:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v22, 0x400, v36
	v_cmp_lt_u32_e64 s[12:13], s30, v22
                                        ; implicit-def: $vgpr59
	s_and_saveexec_b64 s[22:23], s[12:13]
	s_xor_b64 s[12:13], exec, s[22:23]
	s_cbranch_execz .LBB4_67
; %bb.66:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v23, v79
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	global_load_dwordx4 v[58:61], v[22:23], off
.LBB4_67:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_69
; %bb.68:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v22, s29, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[58:61], v22 offset:2048
.LBB4_69:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v22, 0x600, v86
	v_cmp_gt_u32_e64 s[12:13], s2, v22
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_perm_b32 v47, v49, v61, s33
	v_perm_b32 v111, v48, v60, s33
	v_bfi_b32 v107, s31, v59, v17
	v_alignbit_b32 v48, v69, v61, 16
	v_alignbit_b32 v113, v67, v60, 16
	v_alignbit_b32 v108, v66, v59, 16
	v_perm_b32 v49, v19, v41, s33
	v_perm_b32 v112, v18, v40, s33
	v_bfi_b32 v59, s31, v39, v90
	v_alignbit_b32 v41, v65, v41, 16
	v_alignbit_b32 v40, v64, v40, 16
	v_alignbit_b32 v39, v63, v39, 16
	s_and_saveexec_b64 s[22:23], s[12:13]
	s_cbranch_execz .LBB4_87
; %bb.70:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v78, 0x600, v78
	v_lshl_add_u64 v[0:1], v[78:79], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	v_cmp_lt_u32_e64 s[12:13], s30, v22
                                        ; implicit-def: $vgpr64
                                        ; implicit-def: $vgpr85
                                        ; implicit-def: $vgpr120
                                        ; implicit-def: $vgpr121
                                        ; implicit-def: $vgpr122
	s_and_saveexec_b64 s[36:37], s[12:13]
	s_xor_b64 s[12:13], exec, s[36:37]
	s_cbranch_execz .LBB4_72
; %bb.71:                               ;   in Loop: Header=BB4_18 Depth=2
	v_mov_b32_e32 v23, v79
	v_lshl_add_u64 v[18:19], v[22:23], 1, s[6:7]
	global_load_dwordx4 v[64:67], v[18:19], off
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v85, 16, v64
	v_lshrrev_b32_e32 v120, 16, v65
	v_lshrrev_b32_e32 v121, 16, v66
	v_lshrrev_b32_e32 v122, 16, v67
.LBB4_72:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_74
; %bb.73:                               ;   in Loop: Header=BB4_18 Depth=2
	ds_read_b128 v[64:67], v73 offset:3072
	s_waitcnt lgkmcnt(0)
	v_lshrrev_b32_e32 v85, 16, v64
	v_lshrrev_b32_e32 v120, 16, v65
	v_lshrrev_b32_e32 v121, 16, v66
	v_lshrrev_b32_e32 v122, 16, v67
.LBB4_74:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v78, 0x600, v68
	v_cmp_lt_u32_e64 s[12:13], s30, v78
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[36:37], s[12:13]
	s_xor_b64 s[12:13], exec, s[36:37]
	s_cbranch_execz .LBB4_76
; %bb.75:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshl_add_u64 v[18:19], v[78:79], 1, s[6:7]
	global_load_dwordx4 v[68:71], v[18:19], off
.LBB4_76:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_78
; %bb.77:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s28, v73
	s_waitcnt vmcnt(0)
	ds_read_b128 v[68:71], v18 offset:3072
.LBB4_78:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v78, 0x600, v62
	v_cmp_lt_u32_e64 s[12:13], s30, v78
                                        ; implicit-def: $vgpr60
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr23
	s_and_saveexec_b64 s[36:37], s[12:13]
	s_xor_b64 s[12:13], exec, s[36:37]
	s_cbranch_execz .LBB4_80
; %bb.79:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshl_add_u64 v[18:19], v[78:79], 1, s[6:7]
	global_load_dwordx4 v[60:63], v[18:19], off
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v18, 16, v60
	v_lshrrev_b32_e32 v19, 16, v61
	v_lshrrev_b32_e32 v22, 16, v62
	v_lshrrev_b32_e32 v23, 16, v63
.LBB4_80:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_82
; %bb.81:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v18, s24, v73
	ds_read_b128 v[60:63], v18 offset:3072
	s_waitcnt lgkmcnt(0)
	v_lshrrev_b32_e32 v18, 16, v60
	v_lshrrev_b32_e32 v19, 16, v61
	v_lshrrev_b32_e32 v22, 16, v62
	v_lshrrev_b32_e32 v23, 16, v63
.LBB4_82:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v78, 0x600, v36
	v_cmp_lt_u32_e64 s[12:13], s30, v78
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_perm_b32 v96, v67, v71, s33
	v_perm_b32 v97, v66, v70, s33
	v_perm_b32 v102, v65, v69, s33
	v_perm_b32 v101, v64, v68, s33
	v_alignbit_b32 v98, v122, v71, 16
	v_alignbit_b32 v99, v121, v70, 16
	v_alignbit_b32 v106, v120, v69, 16
	v_alignbit_b32 v105, v85, v68, 16
                                        ; implicit-def: $vgpr117
                                        ; implicit-def: $vgpr119
                                        ; implicit-def: $vgpr115
                                        ; implicit-def: $vgpr114
                                        ; implicit-def: $vgpr116
                                        ; implicit-def: $vgpr118
                                        ; implicit-def: $vgpr110
                                        ; implicit-def: $vgpr109
	s_and_saveexec_b64 s[36:37], s[12:13]
	s_xor_b64 s[12:13], exec, s[36:37]
	s_cbranch_execz .LBB4_84
; %bb.83:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshl_add_u64 v[96:97], v[78:79], 1, s[6:7]
	global_load_dwordx4 v[124:127], v[96:97], off
	v_perm_b32 v96, v67, v71, s33
	v_perm_b32 v97, v66, v70, s33
	v_perm_b32 v102, v65, v69, s33
	v_perm_b32 v101, v64, v68, s33
	v_alignbit_b32 v98, v122, v71, 16
	v_alignbit_b32 v99, v121, v70, 16
	v_alignbit_b32 v106, v120, v69, 16
	s_waitcnt vmcnt(0)
	v_perm_b32 v109, v63, v127, s33
	v_perm_b32 v110, v62, v126, s33
	v_perm_b32 v118, v61, v125, s33
	v_perm_b32 v116, v60, v124, s33
	v_alignbit_b32 v114, v23, v127, 16
	v_alignbit_b32 v115, v22, v126, 16
	v_alignbit_b32 v119, v19, v125, 16
	v_alignbit_b32 v117, v18, v124, 16
	v_alignbit_b32 v105, v85, v68, 16
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr22
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr18
.LBB4_84:                               ;   in Loop: Header=BB4_18 Depth=2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_86
; %bb.85:                               ;   in Loop: Header=BB4_18 Depth=2
	v_add_u32_e32 v36, s29, v73
	ds_read_b128 v[64:67], v36 offset:3072
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v109, v63, v67, s33
	v_perm_b32 v110, v62, v66, s33
	v_perm_b32 v118, v61, v65, s33
	v_perm_b32 v116, v60, v64, s33
	v_alignbit_b32 v114, v23, v67, 16
	v_alignbit_b32 v115, v22, v66, 16
	v_alignbit_b32 v119, v19, v65, 16
	v_alignbit_b32 v117, v18, v64, 16
.LBB4_86:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[12:13]
.LBB4_87:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB4_88:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB4_89:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB4_90:                               ;   in Loop: Header=BB4_18 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB4_17
; %bb.91:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshlrev_b32_e32 v18, 16, v12
	v_and_b32_e32 v22, 0xffff0000, v12
	v_lshlrev_b32_e32 v61, 16, v24
	v_lshlrev_b32_e32 v63, 16, v87
	v_lshlrev_b32_e32 v36, 16, v13
	v_and_b32_e32 v64, 0xffff0000, v13
	v_lshlrev_b32_e32 v66, 16, v14
	v_and_b32_e32 v68, 0xffff0000, v14
	v_lshlrev_b32_e32 v70, 16, v15
	v_and_b32_e32 v78, 0xffff0000, v15
	v_lshlrev_b32_e32 v60, 16, v26
	v_and_b32_e32 v62, 0xffff0000, v26
	v_pk_mul_f32 v[60:61], v[18:19], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v121, 0xffff0000, v87
	v_lshlrev_b32_e32 v120, 16, v27
	v_pk_fma_f32 v[60:61], v[120:121], v[36:37], v[60:61] op_sel_hi:[1,0,1]
	v_and_b32_e32 v120, 0xffff0000, v27
	v_lshlrev_b32_e32 v121, 16, v91
	v_pk_fma_f32 v[62:63], v[120:121], v[64:65], v[62:63] op_sel_hi:[1,0,1]
	v_and_b32_e32 v121, 0xffff0000, v91
	v_lshlrev_b32_e32 v120, 16, v28
	v_pk_fma_f32 v[60:61], v[120:121], v[66:67], v[60:61] op_sel_hi:[1,0,1]
	v_and_b32_e32 v120, 0xffff0000, v28
	v_lshlrev_b32_e32 v121, 16, v25
	v_pk_fma_f32 v[62:63], v[120:121], v[68:69], v[62:63] op_sel_hi:[1,0,1]
	v_and_b32_e32 v121, 0xffff0000, v37
	v_lshlrev_b32_e32 v120, 16, v37
	v_pk_fma_f32 v[60:61], v[120:121], v[70:71], v[60:61] op_sel_hi:[1,0,1]
	v_and_b32_e32 v121, 0xffff0000, v29
	v_lshlrev_b32_e32 v120, 16, v29
	v_pk_fma_f32 v[62:63], v[120:121], v[78:79], v[62:63] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[60:61], v[60:61], v[62:63]
	s_nop 0
	v_pk_add_f32 v[82:83], v[60:61], v[82:83]
	v_lshlrev_b32_e32 v61, 16, v34
	v_lshlrev_b32_e32 v63, 16, v93
	v_lshlrev_b32_e32 v60, 16, v50
	v_and_b32_e32 v62, 0xffff0000, v50
	v_pk_mul_f32 v[18:19], v[18:19], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[62:63] op_sel_hi:[0,1]
	v_and_b32_e32 v61, 0xffff0000, v93
	v_lshlrev_b32_e32 v60, 16, v51
	v_pk_fma_f32 v[18:19], v[60:61], v[36:37], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v60, 0xffff0000, v51
	v_lshlrev_b32_e32 v61, 16, v94
	v_pk_fma_f32 v[22:23], v[60:61], v[64:65], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v94
	v_lshlrev_b32_e32 v60, 16, v52
	v_pk_fma_f32 v[18:19], v[60:61], v[66:67], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v60, 0xffff0000, v52
	v_lshlrev_b32_e32 v61, 16, v35
	v_pk_fma_f32 v[22:23], v[60:61], v[68:69], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v100
	v_lshlrev_b32_e32 v60, 16, v100
	v_pk_fma_f32 v[18:19], v[60:61], v[70:71], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v53
	v_lshlrev_b32_e32 v60, 16, v53
	v_pk_fma_f32 v[22:23], v[60:61], v[78:79], v[22:23] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_pk_add_f32 v[80:81], v[18:19], v[80:81]
	v_add_u32_e32 v18, 0x200, v86
	v_cmp_gt_u32_e32 vcc, s2, v18
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_16
; %bb.92:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshlrev_b32_e32 v18, 16, v8
	v_and_b32_e32 v22, 0xffff0000, v8
	v_lshlrev_b32_e32 v36, 16, v9
	v_and_b32_e32 v60, 0xffff0000, v9
	v_lshlrev_b32_e32 v62, 16, v10
	v_and_b32_e32 v64, 0xffff0000, v10
	v_lshlrev_b32_e32 v66, 16, v11
	v_and_b32_e32 v68, 0xffff0000, v11
	v_lshlrev_b32_e32 v71, 16, v20
	v_lshlrev_b32_e32 v70, 16, v30
	v_pk_mul_f32 v[70:71], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v120, 0xffff0000, v30
	v_lshlrev_b32_e32 v121, 16, v92
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121] op_sel_hi:[0,1]
	v_and_b32_e32 v123, 0xffff0000, v92
	v_lshlrev_b32_e32 v122, 16, v31
	v_pk_fma_f32 v[70:71], v[122:123], v[36:37], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v122, 0xffff0000, v31
	v_lshlrev_b32_e32 v123, 16, v21
	v_pk_fma_f32 v[120:121], v[122:123], v[60:61], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v44
	v_lshlrev_b32_e32 v122, 16, v44
	v_pk_fma_f32 v[70:71], v[122:123], v[62:63], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v32
	v_lshlrev_b32_e32 v122, 16, v32
	v_pk_fma_f32 v[120:121], v[122:123], v[64:65], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v57
	v_lshlrev_b32_e32 v122, 16, v57
	v_pk_fma_f32 v[70:71], v[122:123], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v33
	v_lshlrev_b32_e32 v122, 16, v33
	v_pk_fma_f32 v[120:121], v[122:123], v[68:69], v[120:121] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[70:71], v[70:71], v[120:121]
	s_nop 0
	v_pk_add_f32 v[82:83], v[70:71], v[82:83]
	v_lshlrev_b32_e32 v71, 16, v42
	v_lshlrev_b32_e32 v70, 16, v54
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v70, 0xffff0000, v54
	v_lshlrev_b32_e32 v71, 16, v95
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v95
	v_lshlrev_b32_e32 v70, 16, v55
	v_pk_fma_f32 v[18:19], v[70:71], v[36:37], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v70, 0xffff0000, v55
	v_lshlrev_b32_e32 v71, 16, v43
	v_pk_fma_f32 v[22:23], v[70:71], v[60:61], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v45
	v_lshlrev_b32_e32 v60, 16, v45
	v_pk_fma_f32 v[18:19], v[60:61], v[62:63], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v56
	v_lshlrev_b32_e32 v60, 16, v56
	v_pk_fma_f32 v[22:23], v[60:61], v[64:65], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v103
	v_lshlrev_b32_e32 v60, 16, v103
	v_pk_fma_f32 v[18:19], v[60:61], v[66:67], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v104
	v_lshlrev_b32_e32 v60, 16, v104
	v_pk_fma_f32 v[22:23], v[60:61], v[68:69], v[22:23] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_pk_add_f32 v[80:81], v[18:19], v[80:81]
	v_add_u32_e32 v18, 0x400, v86
	v_cmp_gt_u32_e32 vcc, s2, v18
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB4_15
; %bb.93:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshlrev_b32_e32 v18, 16, v4
	v_and_b32_e32 v22, 0xffff0000, v4
	v_lshlrev_b32_e32 v36, 16, v5
	v_and_b32_e32 v60, 0xffff0000, v5
	v_lshlrev_b32_e32 v62, 16, v6
	v_and_b32_e32 v64, 0xffff0000, v6
	v_lshlrev_b32_e32 v66, 16, v7
	v_and_b32_e32 v68, 0xffff0000, v7
	v_lshlrev_b32_e32 v71, 16, v16
	v_lshlrev_b32_e32 v70, 16, v38
	v_pk_mul_f32 v[70:71], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v120, 0xffff0000, v38
	v_lshlrev_b32_e32 v121, 16, v90
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121] op_sel_hi:[0,1]
	v_and_b32_e32 v123, 0xffff0000, v59
	v_lshlrev_b32_e32 v122, 16, v59
	v_pk_fma_f32 v[70:71], v[122:123], v[36:37], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v39
	v_lshlrev_b32_e32 v122, 16, v39
	v_pk_fma_f32 v[120:121], v[122:123], v[60:61], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v112
	v_lshlrev_b32_e32 v122, 16, v112
	v_pk_fma_f32 v[70:71], v[122:123], v[62:63], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v40
	v_lshlrev_b32_e32 v122, 16, v40
	v_pk_fma_f32 v[120:121], v[122:123], v[64:65], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v49
	v_lshlrev_b32_e32 v122, 16, v49
	v_pk_fma_f32 v[70:71], v[122:123], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v41
	v_lshlrev_b32_e32 v122, 16, v41
	v_pk_fma_f32 v[120:121], v[122:123], v[68:69], v[120:121] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[70:71], v[70:71], v[120:121]
	s_nop 0
	v_pk_add_f32 v[82:83], v[70:71], v[82:83]
	v_lshlrev_b32_e32 v71, 16, v46
	v_lshlrev_b32_e32 v70, 16, v58
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v70, 0xffff0000, v58
	v_lshlrev_b32_e32 v71, 16, v17
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v107
	v_lshlrev_b32_e32 v70, 16, v107
	v_pk_fma_f32 v[18:19], v[70:71], v[36:37], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v71, 0xffff0000, v108
	v_lshlrev_b32_e32 v70, 16, v108
	v_pk_fma_f32 v[22:23], v[70:71], v[60:61], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v111
	v_lshlrev_b32_e32 v60, 16, v111
	v_pk_fma_f32 v[18:19], v[60:61], v[62:63], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v113
	v_lshlrev_b32_e32 v60, 16, v113
	v_pk_fma_f32 v[22:23], v[60:61], v[64:65], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v47
	v_lshlrev_b32_e32 v60, 16, v47
	v_pk_fma_f32 v[18:19], v[60:61], v[66:67], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v48
	v_lshlrev_b32_e32 v60, 16, v48
	v_pk_fma_f32 v[22:23], v[60:61], v[68:69], v[22:23] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_pk_add_f32 v[80:81], v[18:19], v[80:81]
	v_add_u32_e32 v18, 0x600, v86
	v_cmp_gt_u32_e32 vcc, s2, v18
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB4_14
; %bb.94:                               ;   in Loop: Header=BB4_18 Depth=2
	v_lshlrev_b32_e32 v18, 16, v0
	v_and_b32_e32 v22, 0xffff0000, v0
	v_lshlrev_b32_e32 v36, 16, v1
	v_and_b32_e32 v60, 0xffff0000, v1
	v_lshlrev_b32_e32 v62, 16, v2
	v_and_b32_e32 v64, 0xffff0000, v2
	v_lshlrev_b32_e32 v66, 16, v3
	v_and_b32_e32 v68, 0xffff0000, v3
	v_and_b32_e32 v71, 0xffff0000, v101
	v_lshlrev_b32_e32 v70, 16, v101
	v_pk_mul_f32 v[70:71], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v121, 0xffff0000, v105
	v_lshlrev_b32_e32 v120, 16, v105
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121] op_sel_hi:[0,1]
	v_and_b32_e32 v123, 0xffff0000, v102
	v_lshlrev_b32_e32 v122, 16, v102
	v_pk_fma_f32 v[70:71], v[122:123], v[36:37], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v106
	v_lshlrev_b32_e32 v122, 16, v106
	v_pk_fma_f32 v[120:121], v[122:123], v[60:61], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v97
	v_lshlrev_b32_e32 v122, 16, v97
	v_pk_fma_f32 v[70:71], v[122:123], v[62:63], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v99
	v_lshlrev_b32_e32 v122, 16, v99
	v_pk_fma_f32 v[120:121], v[122:123], v[64:65], v[120:121] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v96
	v_lshlrev_b32_e32 v122, 16, v96
	v_pk_fma_f32 v[70:71], v[122:123], v[66:67], v[70:71] op_sel_hi:[1,0,1]
	v_and_b32_e32 v123, 0xffff0000, v98
	v_lshlrev_b32_e32 v122, 16, v98
	v_pk_fma_f32 v[120:121], v[122:123], v[68:69], v[120:121] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[70:71], v[70:71], v[120:121]
	s_nop 0
	v_pk_add_f32 v[82:83], v[70:71], v[82:83]
	v_and_b32_e32 v71, 0xffff0000, v116
	v_lshlrev_b32_e32 v70, 16, v116
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v117
	v_lshlrev_b32_e32 v70, 16, v117
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] op_sel_hi:[0,1]
	v_and_b32_e32 v71, 0xffff0000, v118
	v_lshlrev_b32_e32 v70, 16, v118
	v_pk_fma_f32 v[18:19], v[70:71], v[36:37], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v71, 0xffff0000, v119
	v_lshlrev_b32_e32 v70, 16, v119
	v_pk_fma_f32 v[22:23], v[70:71], v[60:61], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v110
	v_lshlrev_b32_e32 v60, 16, v110
	v_pk_fma_f32 v[18:19], v[60:61], v[62:63], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v115
	v_lshlrev_b32_e32 v60, 16, v115
	v_pk_fma_f32 v[22:23], v[60:61], v[64:65], v[22:23] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v109
	v_lshlrev_b32_e32 v60, 16, v109
	v_pk_fma_f32 v[18:19], v[60:61], v[66:67], v[18:19] op_sel_hi:[1,0,1]
	v_and_b32_e32 v61, 0xffff0000, v114
	v_lshlrev_b32_e32 v60, 16, v114
	v_pk_fma_f32 v[22:23], v[60:61], v[68:69], v[22:23] op_sel_hi:[1,0,1]
	s_nop 0
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_pk_add_f32 v[80:81], v[18:19], v[80:81]
	s_branch .LBB4_14
.LBB4_95:                               ;   in Loop: Header=BB4_12 Depth=1
	v_lshrrev_b32_e32 v18, 16, v41
	v_lshrrev_b32_e32 v19, 16, v49
	v_lshrrev_b32_e32 v22, 16, v99
	v_lshrrev_b32_e32 v23, 16, v97
	v_lshrrev_b32_e32 v36, 16, v48
	v_lshrrev_b32_e32 v60, 16, v47
	v_lshrrev_b32_e32 v61, 16, v115
	v_lshrrev_b32_e32 v62, 16, v110
	s_branch .LBB4_97
.LBB4_96:                               ;   in Loop: Header=BB4_12 Depth=1
	v_lshrrev_b32_e32 v62, 16, v110
	v_lshrrev_b32_e32 v61, 16, v115
	v_lshrrev_b32_e32 v60, 16, v47
	v_lshrrev_b32_e32 v36, 16, v48
	v_lshrrev_b32_e32 v23, 16, v97
	v_lshrrev_b32_e32 v22, 16, v99
	v_lshrrev_b32_e32 v19, 16, v49
	v_lshrrev_b32_e32 v18, 16, v41
	v_mov_b32_e32 v83, v79
	v_mov_b32_e32 v82, v79
	v_mov_b32_e32 v81, v79
	v_mov_b32_e32 v80, v79
.LBB4_97:                               ;   in Loop: Header=BB4_12 Depth=1
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
	v_add_f32 v82, v82, v82 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v82, v82, v82 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v82, v82, v82 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v82, v82, v82 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v82, v82, v82 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v82, v82, v82 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v81, v81, v81 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v80, v80, v80 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB4_11
; %bb.98:                               ;   in Loop: Header=BB4_12 Depth=1
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB4_11
; %bb.99:                               ;   in Loop: Header=BB4_12 Depth=1
	v_and_b32_e32 v63, 0x7f800000, v83
	v_cmp_ne_u32_e32 vcc, s34, v63
                                        ; implicit-def: $vgpr63
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.100:                              ;   in Loop: Header=BB4_12 Depth=1
	v_bfe_u32 v63, v83, 16, 1
	v_add3_u32 v63, v83, v63, s30
; %bb.101:                              ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.102:                              ;   in Loop: Header=BB4_12 Depth=1
	v_or_b32_e32 v63, 0x10000, v83
	v_cmp_eq_u32_sdwa vcc, v83, v79 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v83, vcc
; %bb.103:                              ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v73, v79
	v_lshl_add_u64 v[64:65], v[72:73], 1, s[8:9]
	global_store_short_d16_hi v[64:65], v63, off
	v_and_b32_e32 v63, 0x7f800000, v82
	v_cmp_ne_u32_e32 vcc, s34, v63
                                        ; implicit-def: $vgpr63
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.104:                              ;   in Loop: Header=BB4_12 Depth=1
	v_bfe_u32 v63, v82, 16, 1
	v_add3_u32 v63, v82, v63, s30
                                        ; implicit-def: $vgpr82
; %bb.105:                              ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.106:                              ;   in Loop: Header=BB4_12 Depth=1
	v_or_b32_e32 v63, 0x10000, v82
	v_cmp_eq_u32_sdwa vcc, v82, v79 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v82, vcc
; %bb.107:                              ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v78, s3, v72
	v_lshl_add_u64 v[64:65], v[78:79], 1, s[8:9]
	global_store_short_d16_hi v[64:65], v63, off
	v_and_b32_e32 v63, 0x7f800000, v81
	v_cmp_ne_u32_e32 vcc, s34, v63
                                        ; implicit-def: $vgpr63
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.108:                              ;   in Loop: Header=BB4_12 Depth=1
	v_bfe_u32 v63, v81, 16, 1
	v_add3_u32 v63, v81, v63, s30
; %bb.109:                              ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.110:                              ;   in Loop: Header=BB4_12 Depth=1
	v_or_b32_e32 v63, 0x10000, v81
	v_cmp_eq_u32_sdwa vcc, v81, v79 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v81, vcc
; %bb.111:                              ;   in Loop: Header=BB4_12 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v78, s3, v78
	v_lshl_add_u64 v[64:65], v[78:79], 1, s[8:9]
	global_store_short_d16_hi v[64:65], v63, off
	v_and_b32_e32 v63, 0x7f800000, v80
	v_cmp_ne_u32_e32 vcc, s34, v63
                                        ; implicit-def: $vgpr63
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.112:                              ;   in Loop: Header=BB4_12 Depth=1
	v_bfe_u32 v63, v80, 16, 1
	v_add3_u32 v63, v80, v63, s30
                                        ; implicit-def: $vgpr80
; %bb.113:                              ;   in Loop: Header=BB4_12 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB4_10
; %bb.114:                              ;   in Loop: Header=BB4_12 Depth=1
	v_or_b32_e32 v63, 0x10000, v80
	v_cmp_eq_u32_sdwa vcc, v80, v79 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v80, vcc
	s_branch .LBB4_10
.LBB4_115:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
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
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 128
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
	.section	.text._Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end4:
	.size	_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end4-_Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5460
; NumSgprs: 44
; NumVgprs: 128
; NumAgprs: 0
; TotalNumVgprs: 128
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 128
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_b46f700966eaca39,@object ; @__hip_cuid_b46f700966eaca39
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b46f700966eaca39
__hip_cuid_b46f700966eaca39:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b46f700966eaca39, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b46f700966eaca39
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
    .name:           _Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         _Z12wvSplitK_hf_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     105
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
    .name:           _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         _Z12wvSplitK_hf_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
