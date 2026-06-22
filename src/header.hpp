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
    GEMM = 0b1110000,
    TACCGET = 0b1110001,
};

STRUCT(TOPRand) {
    DType dtype;
    Int<VTR_WID> vidx;
    Int<2> vmask;
    bool is_temp;
    bool is_zero;
    bool valid;
};

STRUCT(TInstRaw) {
    SubOPCode subopcode;
    Int<7> funct7;
    Int<96> args;
    TOPRand trd;
    TOPRand trs1;
    TOPRand trs2;
    bool acc_valid;
    bool acc_new;
};

STRUCT(TInst) {
    SubOPCode subopcode;
    Int<5> funct5;
    Int<96> args;
    TOPRand trd;
    TOPRand trs1;
    TOPRand trs2;
    Int<2> mask;
    bool acc_valid;
    bool acc_new;

    Int<PTR_WID*4> ptrd;
    Int<PTR_WID*4> ptrs1;
    Int<PTR_WID*4> ptrs2;
    Int<PTEMP_WID> ptemp;
    Int<PTEMP_WID> ptempd;
    Int<PMASK_WID> pmask;
    Int<PMASK_WID> pmaskd;
    Int<PACC_WID> pacc;
};
