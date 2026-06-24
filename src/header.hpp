#pragma once

#include "defhelper.hpp"

CONFIG(TILE_SIZE, 32);
CONFIG(VTREG_SIZE, 384);
CONFIG(PTREG_SIZE, 512);
CONFIG(PACC_NUM, 8);
CONFIG(PTEMP_NUM, 8);
CONFIG(PMASK_NUM, 6);
CONFIG(SIMD_LANE_SIZE, 8);

CONFIG(GEMM_LANE_SIZE, 4);
CONFIG(GEMM_LAST_LEVEL_LATENCY, 8);
CONFIG(GEMM_OUTPUT_LATENCY, 8);

ALIAS_ARRAY1(TileRow, uint8_t, TILE_SIZE);
ALIAS_ARRAY1(TileSIMDRow, uint8_t, SIMD_LANE_SIZE);

ALIAS_ARRAY1(Tile, TileRow, TILE_SIZE);
ALIAS_ARRAY1(TileSIMD, TileSIMDRow, TILE_SIZE);

ALIAS_ARRAY1(Tile32Row, uint32_t, TILE_SIZE);
ALIAS_ARRAY1(Tile32SIMDRow, uint32_t, SIMD_LANE_SIZE);

ALIAS_ARRAY1(Tile32, Tile32Row, TILE_SIZE);
ALIAS_ARRAY1(Tile32SIMD, Tile32SIMDRow, TILE_SIZE);

CONFIG(VTR_WID, clog2(VTREG_SIZE));
CONFIG(PTR_WID, clog2(PTREG_SIZE));

CONFIG(PACC_WID, clog2(PACC_NUM));
CONFIG(PTEMP_WID, clog2(PTEMP_NUM));
CONFIG(PMASK_WID, clog2(PMASK_NUM));

ENUM(DType) {
    DTYPE_SINT8 = 0x00,
    DTYPE_UINT8 = 0x01,
    DTYPE_F8E4M3 = 0x02,
    DTYPE_F8E5M2 = 0x03,
    DTYPE_SINT16 = 0x10,
    DTYPE_UINT16 = 0x11,
    DTYPE_F16 = 0x12,
    DTYPE_SINT32 = 0x20,
    DTYPE_UINT32 = 0x21,
    DTYPE_F32 = 0x22
};

ENUM(SubOPCode) {
    // 暂时先放GEMM相关的指令，其他指令后续再加
    GEMM = 0x70,
    TACCGET = 0x71,
};

ENUM(TOPRandType) {
    INVALID = 0x00,
    TREG = 0x01,
    TTEMP = 0x02,
    TACC = 0x03,
    TMASK = 0x04,
};

STRUCT(TInstRaw) {
    Int<8> subopcode;
    Int<96> args;
    Int<5> funct5;
    Int<2> mask;

    DType rd_datatype;
    DType rs1_datatype;
    DType rs2_datatype;

    TOPRandType rd_type;
    TOPRandType rs1_type;
    TOPRandType rs2_type;

    Int<VTR_WID> rd_vidx;
    Int<VTR_WID> rs1_vidx;
    Int<VTR_WID> rs2_vidx;
};

STRUCT(TInst) {
    Int<8> subopcode;
    Int<96> args;
    Int<5> funct5;
    Int<2> mask;

    DType rd_datatype;
    DType rs1_datatype;
    DType rs2_datatype;

    TOPRandType rd_type;
    TOPRandType rs1_type;
    TOPRandType rs2_type;

    Int<VTR_WID> rd_vidx;
    Int<VTR_WID> rs1_vidx;
    Int<VTR_WID> rs2_vidx;

    Int<PTR_WID> rd_pidx[4];
    Int<PTR_WID> rs1_pidx[4];
    Int<PTR_WID> rs2_pidx[4];

    Int<PMASK_WID> pmask;
};

STRUCT(RNMMainInput) {
    Int<VTR_WID> vtrs[8];
    Int<VTR_WID> vtrd[4];
    Int<8> vtrs_valid;
    Int<4> vtrd_valid;
};

STRUCT(RNMMainOutput) {
    Int<PTR_WID> vtrs[8];
    Int<PTR_WID> vtrd[4];
};

STRUCT(RenameRetireRead) {
    Int<PTR_WID> pidx[8];
    Int<8> valid;
};

STRUCT(RenameRetireWrite) {
    Int<PTR_WID> pidx[4];
    Int<4> valid;
};
