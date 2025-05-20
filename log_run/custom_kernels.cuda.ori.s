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
	v_lshlrev_b32_e32 v64, 3, v2
	s_lshl_b32 s22, s2, 2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_min_i32 s20, s22, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v64
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
	s_cselect_b64 s[24:25], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[24:25], s[24:25], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[24:25]
	s_branch .LBB1_2
.LBB1_5:
	s_or_b64 exec, exec, s[0:1]
.LBB1_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_40
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v66, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v66
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_40
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s23, s11, s10
	s_mul_i32 s24, s2, 6
	v_lshlrev_b32_e32 v65, 4, v2
	s_lshl_b32 s25, s2, 1
	v_mad_u64_u32 v[68:69], s[6:7], s2, v66, v[64:65]
	s_mul_i32 s26, s23, s2
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v71, 0
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr46_vgpr47
                                        ; implicit-def: $vgpr80_vgpr81
                                        ; implicit-def: $vgpr82_vgpr83
                                        ; implicit-def: $vgpr54_vgpr55
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $vgpr26_vgpr27
                                        ; implicit-def: $vgpr72_vgpr73
                                        ; implicit-def: $vgpr74_vgpr75
                                        ; implicit-def: $vgpr30_vgpr31
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr38_vgpr39
                                        ; implicit-def: $vgpr76_vgpr77
                                        ; implicit-def: $vgpr78_vgpr79
                                        ; implicit-def: $vgpr42_vgpr43
                                        ; implicit-def: $vgpr34_vgpr35
                                        ; implicit-def: $vgpr62_vgpr63
                                        ; implicit-def: $vgpr86_vgpr87
                                        ; implicit-def: $vgpr84_vgpr85
                                        ; implicit-def: $vgpr58_vgpr59
                                        ; implicit-def: $vgpr50_vgpr51
	s_branch .LBB1_10
.LBB1_9:                                ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v66, s23, v66
	v_cmp_le_u32_e32 vcc, s3, v66
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v68, s26, v68
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB1_40
.LBB1_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_16 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_mov_b32 s27, 0
	s_cbranch_vccnz .LBB1_37
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v90, v65
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v67, 0
	s_branch .LBB1_16
.LBB1_12:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_13:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_14:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB1_15:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s27, 0x800
	s_cmp_ge_u32 s27, s2
	v_add_u32_e32 v90, 0x1000, v90
	s_cbranch_scc1 .LBB1_38
.LBB1_16:                               ;   Parent Loop BB1_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v91, s27, v64
	v_cmp_gt_u32_e32 vcc, s2, v91
	v_add_u32_e32 v92, 0x200, v91
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execnz .LBB1_19
; %bb.17:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execnz .LBB1_26
.LBB1_18:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_15
	s_branch .LBB1_33
.LBB1_19:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v70, s27, v68
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[70:71], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	v_cmp_gt_u32_e64 s[10:11], s2, v92
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB1_25
; %bb.20:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v8, 0x200, v70
	v_mov_b32_e32 v9, v71
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	v_add_u32_e32 v93, 0x400, v91
	v_cmp_gt_u32_e64 s[10:11], s2, v93
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB1_24
; %bb.21:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v4, 0x400, v70
	v_mov_b32_e32 v5, v71
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	v_add_u32_e32 v93, 0x600, v91
	v_cmp_gt_u32_e64 s[10:11], s2, v93
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB1_23
; %bb.22:                               ;   in Loop: Header=BB1_16 Depth=2
	v_add_u32_e32 v70, 0x600, v70
	v_lshl_add_u64 v[0:1], v[70:71], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
.LBB1_23:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB1_24:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_25:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_18
.LBB1_26:                               ;   in Loop: Header=BB1_16 Depth=2
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[48:51], v90
	v_add_u32_e32 v70, s25, v90
	v_add_u32_e32 v93, s22, v90
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v70
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[84:85], v93 offset1:1
	v_add_u32_e32 v94, s24, v90
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[86:87], v93 offset0:2 offset1:3
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[60:63], v94
	v_cmp_gt_u32_e64 s[10:11], s2, v92
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB1_32
; %bb.27:                               ;   in Loop: Header=BB1_16 Depth=2
	ds_read_b128 v[32:35], v90 offset:1024
	v_add_u32_e32 v36, 0x400, v93
	v_add_u32_e32 v37, 0x408, v93
	ds_read2_b32 v[78:79], v36 offset1:1
	ds_read2_b32 v[76:77], v37 offset1:1
	ds_read_b128 v[40:43], v70 offset:1024
	ds_read_b128 v[36:39], v94 offset:1024
	v_add_u32_e32 v95, 0x400, v91
	v_cmp_gt_u32_e64 s[10:11], s2, v95
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB1_31
; %bb.28:                               ;   in Loop: Header=BB1_16 Depth=2
	ds_read_b128 v[20:23], v90 offset:2048
	v_add_u32_e32 v24, 0x800, v93
	v_add_u32_e32 v25, 0x808, v93
	ds_read2_b32 v[74:75], v24 offset1:1
	ds_read2_b32 v[72:73], v25 offset1:1
	ds_read_b128 v[28:31], v70 offset:2048
	ds_read_b128 v[24:27], v94 offset:2048
	v_add_u32_e32 v95, 0x600, v91
	v_cmp_gt_u32_e64 s[10:11], s2, v95
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB1_30
; %bb.29:                               ;   in Loop: Header=BB1_16 Depth=2
	ds_read_b128 v[16:19], v90 offset:3072
	v_add_u32_e32 v44, 0xc00, v93
	v_add_u32_e32 v45, 0xc08, v93
	ds_read2_b32 v[82:83], v44 offset1:1
	ds_read2_b32 v[80:81], v45 offset1:1
	ds_read_b128 v[52:55], v70 offset:3072
	ds_read_b128 v[44:47], v94 offset:3072
.LBB1_30:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB1_31:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_32:                               ;   in Loop: Header=BB1_16 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_15
.LBB1_33:                               ;   in Loop: Header=BB1_16 Depth=2
	s_waitcnt vmcnt(0) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v48, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v49, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v50, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v51, v15
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v56, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v57, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v58, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v59, v15
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v84, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v85, v13
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v86, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v87, v15
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v60, v12
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v61, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v62, v14
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v63, v15
	;;#ASMEND
	v_cmp_gt_u32_e32 vcc, s2, v92
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_14
; %bb.34:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v32, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v33, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v34, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v35, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v40, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v41, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v42, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v43, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v78, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v79, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v76, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v77, v11
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v36, v8
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v37, v9
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v38, v10
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v39, v11
	;;#ASMEND
	v_add_u32_e32 v70, 0x400, v91
	v_cmp_gt_u32_e32 vcc, s2, v70
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_13
; %bb.35:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v20, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v21, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v22, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v23, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v28, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v29, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v30, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v31, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v74, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v75, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v72, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v73, v7
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v24, v4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v25, v5
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v26, v6
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v27, v7
	;;#ASMEND
	v_add_u32_e32 v70, 0x600, v91
	v_cmp_gt_u32_e32 vcc, s2, v70
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_12
; %bb.36:                               ;   in Loop: Header=BB1_16 Depth=2
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v16, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v17, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v18, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v67, v19, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v52, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v53, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v54, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v89, v55, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v82, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v83, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v80, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v88, v81, v3
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v44, v0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v45, v1
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v46, v2
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v69, v47, v3
	;;#ASMEND
	s_branch .LBB1_12
.LBB1_37:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v67, v71
	v_mov_b32_e32 v89, v71
	v_mov_b32_e32 v88, v71
	v_mov_b32_e32 v69, v71
.LBB1_38:                               ;   in Loop: Header=BB1_10 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v67, v67, v67 row_bcast:31 bound_ctrl:0
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
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v69, v69, v69 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB1_9
; %bb.39:                               ;   in Loop: Header=BB1_10 Depth=1
	v_cvt_f16_f32_e32 v70, v67
	v_mov_b32_e32 v67, v71
	v_cvt_f16_f32_e32 v89, v89
	v_lshl_add_u64 v[90:91], v[66:67], 1, s[8:9]
	global_store_short v[90:91], v70, off
	v_add_u32_e32 v70, s3, v66
	v_lshl_add_u64 v[90:91], v[70:71], 1, s[8:9]
	global_store_short v[90:91], v89, off
	v_cvt_f16_f32_e32 v67, v88
	v_add_u32_e32 v70, s3, v70
	v_lshl_add_u64 v[88:89], v[70:71], 1, s[8:9]
	v_cvt_f16_f32_e32 v69, v69
	global_store_short v[88:89], v67, off
	v_add_u32_e32 v70, s3, v70
	v_lshl_add_u64 v[88:89], v[70:71], 1, s[8:9]
	global_store_short v[88:89], v69, off
	s_branch .LBB1_9
.LBB1_40:
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
		.amdhsa_next_free_vgpr 96
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 96
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
; codeLenInByte = 2336
; NumSgprs: 34
; NumVgprs: 96
; NumAgprs: 0
; TotalNumVgprs: 96
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 96
; AccumOffset: 96
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 23
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
	v_lshlrev_b32_e32 v64, 3, v2
	s_lshl_b32 s22, s2, 2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB2_6
; %bb.1:
	s_min_i32 s20, s22, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v64
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
	s_cselect_b64 s[24:25], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[24:25], s[24:25], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[24:25]
	s_branch .LBB2_2
.LBB2_5:
	s_or_b64 exec, exec, s[0:1]
.LBB2_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_56
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v66, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v66
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_56
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s23, s11, s10
	s_mul_i32 s24, s2, 6
	v_lshlrev_b32_e32 v65, 4, v2
	s_lshl_b32 s25, s2, 1
	v_mad_u64_u32 v[68:69], s[6:7], s2, v66, v[64:65]
	s_mul_i32 s26, s23, s2
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s27, 0x7f800000
	s_movk_i32 s28, 0x7fff
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr73
                                        ; implicit-def: $vgpr75
                                        ; implicit-def: $vgpr31
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr27
                                        ; implicit-def: $vgpr77
                                        ; implicit-def: $vgpr79
                                        ; implicit-def: $vgpr39
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr43
                                        ; implicit-def: $vgpr81
                                        ; implicit-def: $vgpr83
                                        ; implicit-def: $vgpr51
                                        ; implicit-def: $vgpr47
                                        ; implicit-def: $vgpr55
                                        ; implicit-def: $vgpr87
                                        ; implicit-def: $vgpr85
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr59
	s_branch .LBB2_11
.LBB2_9:                                ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v70, s3, v70
	v_lshl_add_u64 v[88:89], v[70:71], 1, s[8:9]
	global_store_short_d16_hi v[88:89], v67, off
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v66, s23, v66
	v_cmp_le_u32_e32 vcc, s3, v66
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v68, s26, v68
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB2_56
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_17 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB2_38
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	s_mov_b32 s29, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v67, v65
	v_mov_b32_e32 v89, v88
	v_mov_b32_e32 v90, v88
	v_mov_b32_e32 v91, v88
	s_branch .LBB2_17
.LBB2_13:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_14:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB2_15:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB2_16:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s29, 0x800
	s_cmp_ge_u32 s29, s2
	v_add_u32_e32 v67, 0x1000, v67
	s_cbranch_scc1 .LBB2_39
.LBB2_17:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v69, s29, v64
	v_cmp_gt_u32_e32 vcc, s2, v69
	v_add_u32_e32 v92, 0x200, v69
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execnz .LBB2_20
; %bb.18:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execnz .LBB2_27
.LBB2_19:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_16
	s_branch .LBB2_34
.LBB2_20:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v70, s29, v68
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[70:71], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	v_cmp_gt_u32_e64 s[10:11], s2, v92
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB2_26
; %bb.21:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v8, 0x200, v70
	v_mov_b32_e32 v9, v71
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	v_add_u32_e32 v93, 0x400, v69
	v_cmp_gt_u32_e64 s[10:11], s2, v93
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_25
; %bb.22:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v4, 0x400, v70
	v_mov_b32_e32 v5, v71
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	v_add_u32_e32 v93, 0x600, v69
	v_cmp_gt_u32_e64 s[10:11], s2, v93
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB2_24
; %bb.23:                               ;   in Loop: Header=BB2_17 Depth=2
	v_add_u32_e32 v70, 0x600, v70
	v_lshl_add_u64 v[0:1], v[70:71], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
.LBB2_24:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_25:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_26:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_19
.LBB2_27:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v67
	v_add_u32_e32 v70, s25, v67
	v_add_u32_e32 v94, s22, v67
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[60:63], v70
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[84:85], v94 offset1:1
	v_add_u32_e32 v93, s24, v67
	s_waitcnt lgkmcnt(4)
	ds_read2_b32 v[86:87], v94 offset0:2 offset1:3
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[52:55], v93
	v_cmp_gt_u32_e64 s[10:11], s2, v92
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB2_33
; %bb.28:                               ;   in Loop: Header=BB2_17 Depth=2
	ds_read_b128 v[44:47], v67 offset:1024
	v_add_u32_e32 v40, 0x400, v94
	v_add_u32_e32 v41, 0x408, v94
	ds_read2_b32 v[82:83], v40 offset1:1
	ds_read2_b32 v[80:81], v41 offset1:1
	ds_read_b128 v[48:51], v70 offset:1024
	ds_read_b128 v[40:43], v93 offset:1024
	v_add_u32_e32 v95, 0x400, v69
	v_cmp_gt_u32_e64 s[10:11], s2, v95
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_32
; %bb.29:                               ;   in Loop: Header=BB2_17 Depth=2
	ds_read_b128 v[32:35], v67 offset:2048
	v_add_u32_e32 v24, 0x800, v94
	v_add_u32_e32 v25, 0x808, v94
	ds_read2_b32 v[78:79], v24 offset1:1
	ds_read2_b32 v[76:77], v25 offset1:1
	ds_read_b128 v[36:39], v70 offset:2048
	ds_read_b128 v[24:27], v93 offset:2048
	v_add_u32_e32 v95, 0x600, v69
	v_cmp_gt_u32_e64 s[10:11], s2, v95
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB2_31
; %bb.30:                               ;   in Loop: Header=BB2_17 Depth=2
	ds_read_b128 v[16:19], v67 offset:3072
	v_add_u32_e32 v20, 0xc00, v94
	v_add_u32_e32 v21, 0xc08, v94
	ds_read2_b32 v[74:75], v20 offset1:1
	ds_read2_b32 v[72:73], v21 offset1:1
	ds_read_b128 v[28:31], v70 offset:3072
	ds_read_b128 v[20:23], v93 offset:3072
.LBB2_31:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_32:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_33:                               ;   in Loop: Header=BB2_17 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_16
.LBB2_34:                               ;   in Loop: Header=BB2_17 Depth=2
	s_waitcnt lgkmcnt(4)
	v_and_b32_e32 v95, 0xffff0000, v56
	v_lshlrev_b32_e32 v94, 16, v56
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v97, 0xffff0000, v12
	v_lshlrev_b32_e32 v96, 16, v12
	v_pk_mul_f32 v[94:95], v[94:95], v[96:97]
	v_and_b32_e32 v99, 0xffff0000, v57
	v_lshlrev_b32_e32 v98, 16, v57
	v_and_b32_e32 v101, 0xffff0000, v13
	v_lshlrev_b32_e32 v100, 16, v13
	v_pk_mul_f32 v[98:99], v[98:99], v[100:101]
	v_and_b32_e32 v103, 0xffff0000, v58
	v_lshlrev_b32_e32 v102, 16, v58
	v_and_b32_e32 v105, 0xffff0000, v14
	v_lshlrev_b32_e32 v104, 16, v14
	v_pk_mul_f32 v[102:103], v[102:103], v[104:105]
	v_and_b32_e32 v107, 0xffff0000, v59
	v_lshlrev_b32_e32 v106, 16, v59
	v_and_b32_e32 v109, 0xffff0000, v15
	v_lshlrev_b32_e32 v108, 16, v15
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	s_waitcnt lgkmcnt(3)
	v_and_b32_e32 v111, 0xffff0000, v60
	v_lshlrev_b32_e32 v110, 16, v60
	v_pk_mul_f32 v[110:111], v[110:111], v[96:97]
	v_and_b32_e32 v113, 0xffff0000, v61
	v_lshlrev_b32_e32 v112, 16, v61
	v_pk_mul_f32 v[112:113], v[112:113], v[100:101]
	v_and_b32_e32 v115, 0xffff0000, v62
	v_lshlrev_b32_e32 v114, 16, v62
	v_pk_mul_f32 v[114:115], v[114:115], v[104:105]
	v_and_b32_e32 v117, 0xffff0000, v63
	v_lshlrev_b32_e32 v116, 16, v63
	v_pk_mul_f32 v[116:117], v[116:117], v[108:109]
	v_mov_b32_e32 v118, v110
	v_mov_b32_e32 v119, v94
	v_mov_b32_e32 v94, v111
	v_pk_add_f32 v[94:95], v[118:119], v[94:95]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[94:95]
	v_mov_b32_e32 v94, v112
	v_mov_b32_e32 v95, v98
	v_mov_b32_e32 v98, v113
	v_pk_add_f32 v[94:95], v[94:95], v[98:99]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[94:95]
	v_mov_b32_e32 v94, v114
	v_mov_b32_e32 v95, v102
	v_mov_b32_e32 v102, v115
	v_pk_add_f32 v[94:95], v[94:95], v[102:103]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[94:95]
	v_mov_b32_e32 v94, v116
	v_mov_b32_e32 v95, v106
	v_mov_b32_e32 v106, v117
	v_pk_add_f32 v[94:95], v[94:95], v[106:107]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[94:95]
	s_waitcnt lgkmcnt(2)
	v_and_b32_e32 v95, 0xffff0000, v84
	v_lshlrev_b32_e32 v94, 16, v84
	v_pk_mul_f32 v[94:95], v[94:95], v[96:97]
	v_and_b32_e32 v99, 0xffff0000, v85
	v_lshlrev_b32_e32 v98, 16, v85
	v_pk_mul_f32 v[98:99], v[98:99], v[100:101]
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v103, 0xffff0000, v86
	v_lshlrev_b32_e32 v102, 16, v86
	v_pk_mul_f32 v[102:103], v[102:103], v[104:105]
	v_and_b32_e32 v107, 0xffff0000, v87
	v_lshlrev_b32_e32 v106, 16, v87
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v111, 0xffff0000, v52
	v_lshlrev_b32_e32 v110, 16, v52
	v_pk_mul_f32 v[96:97], v[110:111], v[96:97]
	v_and_b32_e32 v111, 0xffff0000, v53
	v_lshlrev_b32_e32 v110, 16, v53
	v_pk_mul_f32 v[100:101], v[110:111], v[100:101]
	v_and_b32_e32 v111, 0xffff0000, v54
	v_lshlrev_b32_e32 v110, 16, v54
	v_pk_mul_f32 v[104:105], v[110:111], v[104:105]
	v_and_b32_e32 v111, 0xffff0000, v55
	v_lshlrev_b32_e32 v110, 16, v55
	v_pk_mul_f32 v[108:109], v[110:111], v[108:109]
	v_mov_b32_e32 v110, v96
	v_mov_b32_e32 v111, v94
	v_mov_b32_e32 v94, v97
	v_pk_add_f32 v[94:95], v[110:111], v[94:95]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[94:95]
	v_mov_b32_e32 v94, v100
	v_mov_b32_e32 v95, v98
	v_mov_b32_e32 v98, v101
	v_pk_add_f32 v[94:95], v[94:95], v[98:99]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[94:95]
	v_mov_b32_e32 v94, v104
	v_mov_b32_e32 v95, v102
	v_mov_b32_e32 v102, v105
	v_pk_add_f32 v[94:95], v[94:95], v[102:103]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[94:95]
	v_mov_b32_e32 v94, v108
	v_mov_b32_e32 v95, v106
	v_mov_b32_e32 v106, v109
	v_pk_add_f32 v[94:95], v[94:95], v[106:107]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[94:95]
	v_cmp_gt_u32_e32 vcc, s2, v92
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_15
; %bb.35:                               ;   in Loop: Header=BB2_17 Depth=2
	v_and_b32_e32 v93, 0xffff0000, v44
	v_lshlrev_b32_e32 v92, 16, v44
	v_and_b32_e32 v95, 0xffff0000, v8
	v_lshlrev_b32_e32 v94, 16, v8
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v45
	v_lshlrev_b32_e32 v96, 16, v45
	v_and_b32_e32 v99, 0xffff0000, v9
	v_lshlrev_b32_e32 v98, 16, v9
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v46
	v_lshlrev_b32_e32 v100, 16, v46
	v_and_b32_e32 v103, 0xffff0000, v10
	v_lshlrev_b32_e32 v102, 16, v10
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v47
	v_lshlrev_b32_e32 v104, 16, v47
	v_and_b32_e32 v107, 0xffff0000, v11
	v_lshlrev_b32_e32 v106, 16, v11
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v48
	v_lshlrev_b32_e32 v108, 16, v48
	v_pk_mul_f32 v[108:109], v[108:109], v[94:95]
	v_and_b32_e32 v111, 0xffff0000, v49
	v_lshlrev_b32_e32 v110, 16, v49
	v_pk_mul_f32 v[110:111], v[110:111], v[98:99]
	v_mov_b32_e32 v112, v108
	v_mov_b32_e32 v113, v92
	v_mov_b32_e32 v92, v109
	v_pk_add_f32 v[92:93], v[112:113], v[92:93]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v110
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v111
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_and_b32_e32 v93, 0xffff0000, v50
	v_lshlrev_b32_e32 v92, 16, v50
	v_pk_mul_f32 v[92:93], v[92:93], v[102:103]
	s_nop 0
	v_mov_b32_e32 v96, v92
	v_mov_b32_e32 v97, v100
	v_mov_b32_e32 v100, v93
	v_pk_add_f32 v[92:93], v[96:97], v[100:101]
	v_and_b32_e32 v97, 0xffff0000, v51
	v_lshlrev_b32_e32 v96, 16, v51
	v_pk_mul_f32 v[96:97], v[96:97], v[106:107]
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v96
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v97
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_and_b32_e32 v93, 0xffff0000, v82
	v_lshlrev_b32_e32 v92, 16, v82
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v83
	v_lshlrev_b32_e32 v96, 16, v83
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v80
	v_lshlrev_b32_e32 v100, 16, v80
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v81
	v_lshlrev_b32_e32 v104, 16, v81
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v40
	v_lshlrev_b32_e32 v108, 16, v40
	v_pk_mul_f32 v[94:95], v[108:109], v[94:95]
	v_and_b32_e32 v109, 0xffff0000, v41
	v_lshlrev_b32_e32 v108, 16, v41
	v_pk_mul_f32 v[98:99], v[108:109], v[98:99]
	v_mov_b32_e32 v108, v94
	v_mov_b32_e32 v109, v92
	v_mov_b32_e32 v92, v95
	v_pk_add_f32 v[92:93], v[108:109], v[92:93]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v98
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v99
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_and_b32_e32 v93, 0xffff0000, v42
	v_lshlrev_b32_e32 v92, 16, v42
	v_pk_mul_f32 v[92:93], v[92:93], v[102:103]
	s_nop 0
	v_mov_b32_e32 v94, v92
	v_mov_b32_e32 v95, v100
	v_mov_b32_e32 v100, v93
	v_pk_add_f32 v[92:93], v[94:95], v[100:101]
	v_and_b32_e32 v95, 0xffff0000, v43
	v_lshlrev_b32_e32 v94, 16, v43
	v_pk_mul_f32 v[94:95], v[94:95], v[106:107]
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v94
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v95
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_add_u32_e32 v70, 0x400, v69
	v_cmp_gt_u32_e32 vcc, s2, v70
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_14
; %bb.36:                               ;   in Loop: Header=BB2_17 Depth=2
	v_and_b32_e32 v93, 0xffff0000, v32
	v_lshlrev_b32_e32 v92, 16, v32
	v_and_b32_e32 v95, 0xffff0000, v4
	v_lshlrev_b32_e32 v94, 16, v4
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v33
	v_lshlrev_b32_e32 v96, 16, v33
	v_and_b32_e32 v99, 0xffff0000, v5
	v_lshlrev_b32_e32 v98, 16, v5
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v34
	v_lshlrev_b32_e32 v100, 16, v34
	v_and_b32_e32 v103, 0xffff0000, v6
	v_lshlrev_b32_e32 v102, 16, v6
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v35
	v_lshlrev_b32_e32 v104, 16, v35
	v_and_b32_e32 v107, 0xffff0000, v7
	v_lshlrev_b32_e32 v106, 16, v7
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v36
	v_lshlrev_b32_e32 v108, 16, v36
	v_pk_mul_f32 v[108:109], v[108:109], v[94:95]
	v_and_b32_e32 v111, 0xffff0000, v37
	v_lshlrev_b32_e32 v110, 16, v37
	v_pk_mul_f32 v[110:111], v[110:111], v[98:99]
	v_and_b32_e32 v113, 0xffff0000, v38
	v_lshlrev_b32_e32 v112, 16, v38
	v_pk_mul_f32 v[112:113], v[112:113], v[102:103]
	v_and_b32_e32 v115, 0xffff0000, v39
	v_lshlrev_b32_e32 v114, 16, v39
	v_pk_mul_f32 v[114:115], v[114:115], v[106:107]
	v_mov_b32_e32 v116, v108
	v_mov_b32_e32 v117, v92
	v_mov_b32_e32 v92, v109
	v_pk_add_f32 v[92:93], v[116:117], v[92:93]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v110
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v111
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v112
	v_mov_b32_e32 v93, v100
	v_mov_b32_e32 v100, v113
	v_pk_add_f32 v[92:93], v[92:93], v[100:101]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v114
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v115
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_and_b32_e32 v93, 0xffff0000, v78
	v_lshlrev_b32_e32 v92, 16, v78
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v79
	v_lshlrev_b32_e32 v96, 16, v79
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v76
	v_lshlrev_b32_e32 v100, 16, v76
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v77
	v_lshlrev_b32_e32 v104, 16, v77
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v24
	v_lshlrev_b32_e32 v108, 16, v24
	v_pk_mul_f32 v[94:95], v[108:109], v[94:95]
	v_and_b32_e32 v109, 0xffff0000, v25
	v_lshlrev_b32_e32 v108, 16, v25
	v_pk_mul_f32 v[98:99], v[108:109], v[98:99]
	v_and_b32_e32 v109, 0xffff0000, v26
	v_lshlrev_b32_e32 v108, 16, v26
	v_pk_mul_f32 v[102:103], v[108:109], v[102:103]
	v_and_b32_e32 v109, 0xffff0000, v27
	v_lshlrev_b32_e32 v108, 16, v27
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	v_mov_b32_e32 v108, v94
	v_mov_b32_e32 v109, v92
	v_mov_b32_e32 v92, v95
	v_pk_add_f32 v[92:93], v[108:109], v[92:93]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v98
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v99
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v102
	v_mov_b32_e32 v93, v100
	v_mov_b32_e32 v100, v103
	v_pk_add_f32 v[92:93], v[92:93], v[100:101]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v106
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v107
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_add_u32_e32 v69, 0x600, v69
	v_cmp_gt_u32_e32 vcc, s2, v69
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_13
; %bb.37:                               ;   in Loop: Header=BB2_17 Depth=2
	v_and_b32_e32 v93, 0xffff0000, v16
	v_lshlrev_b32_e32 v92, 16, v16
	v_and_b32_e32 v95, 0xffff0000, v0
	v_lshlrev_b32_e32 v94, 16, v0
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v17
	v_lshlrev_b32_e32 v96, 16, v17
	v_and_b32_e32 v99, 0xffff0000, v1
	v_lshlrev_b32_e32 v98, 16, v1
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v18
	v_lshlrev_b32_e32 v100, 16, v18
	v_and_b32_e32 v103, 0xffff0000, v2
	v_lshlrev_b32_e32 v102, 16, v2
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v19
	v_lshlrev_b32_e32 v104, 16, v19
	v_and_b32_e32 v107, 0xffff0000, v3
	v_lshlrev_b32_e32 v106, 16, v3
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v28
	v_lshlrev_b32_e32 v108, 16, v28
	v_pk_mul_f32 v[108:109], v[108:109], v[94:95]
	v_and_b32_e32 v111, 0xffff0000, v29
	v_lshlrev_b32_e32 v110, 16, v29
	v_pk_mul_f32 v[110:111], v[110:111], v[98:99]
	v_and_b32_e32 v113, 0xffff0000, v30
	v_lshlrev_b32_e32 v112, 16, v30
	v_pk_mul_f32 v[112:113], v[112:113], v[102:103]
	v_and_b32_e32 v115, 0xffff0000, v31
	v_lshlrev_b32_e32 v114, 16, v31
	v_pk_mul_f32 v[114:115], v[114:115], v[106:107]
	v_mov_b32_e32 v116, v108
	v_mov_b32_e32 v117, v92
	v_mov_b32_e32 v92, v109
	v_pk_add_f32 v[92:93], v[116:117], v[92:93]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v110
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v111
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v112
	v_mov_b32_e32 v93, v100
	v_mov_b32_e32 v100, v113
	v_pk_add_f32 v[92:93], v[92:93], v[100:101]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_mov_b32_e32 v92, v114
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v115
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	v_and_b32_e32 v93, 0xffff0000, v74
	v_lshlrev_b32_e32 v92, 16, v74
	v_pk_mul_f32 v[92:93], v[92:93], v[94:95]
	v_and_b32_e32 v97, 0xffff0000, v75
	v_lshlrev_b32_e32 v96, 16, v75
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	v_and_b32_e32 v101, 0xffff0000, v72
	v_lshlrev_b32_e32 v100, 16, v72
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_and_b32_e32 v105, 0xffff0000, v73
	v_lshlrev_b32_e32 v104, 16, v73
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v20
	v_lshlrev_b32_e32 v108, 16, v20
	v_pk_mul_f32 v[94:95], v[108:109], v[94:95]
	v_and_b32_e32 v109, 0xffff0000, v21
	v_lshlrev_b32_e32 v108, 16, v21
	v_pk_mul_f32 v[98:99], v[108:109], v[98:99]
	v_and_b32_e32 v109, 0xffff0000, v22
	v_lshlrev_b32_e32 v108, 16, v22
	v_pk_mul_f32 v[102:103], v[108:109], v[102:103]
	v_and_b32_e32 v109, 0xffff0000, v23
	v_lshlrev_b32_e32 v108, 16, v23
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	v_mov_b32_e32 v108, v94
	v_mov_b32_e32 v109, v92
	v_mov_b32_e32 v92, v95
	v_pk_add_f32 v[92:93], v[108:109], v[92:93]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v98
	v_mov_b32_e32 v93, v96
	v_mov_b32_e32 v96, v99
	v_pk_add_f32 v[92:93], v[92:93], v[96:97]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v102
	v_mov_b32_e32 v93, v100
	v_mov_b32_e32 v100, v103
	v_pk_add_f32 v[92:93], v[92:93], v[100:101]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	v_mov_b32_e32 v92, v106
	v_mov_b32_e32 v93, v104
	v_mov_b32_e32 v104, v107
	v_pk_add_f32 v[92:93], v[92:93], v[104:105]
	s_nop 0
	v_pk_add_f32 v[88:89], v[88:89], v[92:93]
	s_branch .LBB2_13
.LBB2_38:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mov_b32_e32 v91, v71
	v_mov_b32_e32 v90, v71
	v_mov_b32_e32 v89, v71
	v_mov_b32_e32 v88, v71
.LBB2_39:                               ;   in Loop: Header=BB2_11 Depth=1
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
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v88, v88, v88 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB2_10
; %bb.40:                               ;   in Loop: Header=BB2_11 Depth=1
	v_and_b32_e32 v67, 0x7f800000, v91
	v_cmp_ne_u32_e32 vcc, s27, v67
                                        ; implicit-def: $vgpr69
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.41:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v67, v91, 16, 1
	v_add3_u32 v69, v91, v67, s28
; %bb.42:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.43:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v67, 0x10000, v91
	v_cmp_eq_u32_sdwa vcc, v91, v71 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v91, vcc
; %bb.44:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v67, v71
	v_lshl_add_u64 v[92:93], v[66:67], 1, s[8:9]
	global_store_short_d16_hi v[92:93], v69, off
	v_and_b32_e32 v67, 0x7f800000, v90
	v_cmp_ne_u32_e32 vcc, s27, v67
                                        ; implicit-def: $vgpr67
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.45:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v67, v90, 16, 1
	v_add3_u32 v67, v90, v67, s28
                                        ; implicit-def: $vgpr90
; %bb.46:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.47:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v67, 0x10000, v90
	v_cmp_eq_u32_sdwa vcc, v90, v71 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v67, v67, v90, vcc
; %bb.48:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v70, s3, v66
	v_lshl_add_u64 v[90:91], v[70:71], 1, s[8:9]
	global_store_short_d16_hi v[90:91], v67, off
	v_and_b32_e32 v67, 0x7f800000, v89
	v_cmp_ne_u32_e32 vcc, s27, v67
                                        ; implicit-def: $vgpr67
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.49:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v67, v89, 16, 1
	v_add3_u32 v67, v89, v67, s28
; %bb.50:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.51:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v67, 0x10000, v89
	v_cmp_eq_u32_sdwa vcc, v89, v71 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v67, v67, v89, vcc
; %bb.52:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v70, s3, v70
	v_lshl_add_u64 v[90:91], v[70:71], 1, s[8:9]
	global_store_short_d16_hi v[90:91], v67, off
	v_and_b32_e32 v67, 0x7f800000, v88
	v_cmp_ne_u32_e32 vcc, s27, v67
                                        ; implicit-def: $vgpr67
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.53:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v67, v88, 16, 1
	v_add3_u32 v67, v88, v67, s28
                                        ; implicit-def: $vgpr88
; %bb.54:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB2_9
; %bb.55:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v67, 0x10000, v88
	v_cmp_eq_u32_sdwa vcc, v88, v71 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v67, v67, v88, vcc
	s_branch .LBB2_9
.LBB2_56:
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
		.amdhsa_next_free_vgpr 120
		.amdhsa_next_free_sgpr 30
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
; codeLenInByte = 4408
; NumSgprs: 36
; NumVgprs: 120
; NumAgprs: 0
; TotalNumVgprs: 120
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 36
; NumVGPRsForWavesPerEU: 120
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
	.type	__hip_cuid_54af5210437cd3a5,@object ; @__hip_cuid_54af5210437cd3a5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_54af5210437cd3a5
__hip_cuid_54af5210437cd3a5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_54af5210437cd3a5, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_54af5210437cd3a5
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
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     96
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
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi4ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     120
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
