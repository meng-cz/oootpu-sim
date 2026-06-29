#pragma once

#include "dtype.hpp"
#include "operand.hpp"
#include "tile.hpp"

ENUM(TMiscOp) {
    TFILL = 0x00,
    TFILLLINE = 0x01,
    TBCAST = 0x02,
    TBCAST_R = 0x03,
    TBCAST_C = 0x04,
    TSHIFT = 0x05,
    TSHIFT_Z = 0x06,
    TTRANS = 0x07,
    TTRANS_RR = 0x08,
    TTRANS_RC = 0x09,
    TSETMASK = 0x0a
};

ENUM(TMiscMaskMode) {
    RECT = 0x00,
    UT = 0x01,
    UT_D = 0x02,
    LT = 0x03,
    LT_D = 0x04
};

STRUCT(TMiscInstMeta) {
    TMiscOp op;
    DType rd_datatype;
    DType rs1_datatype;
    TOPRandType rd_type;
    TOPRandType rs1_type;
    Int<PTR_WID> rd_pidx[4];
    Int<PTR_WID> rs1_pidx[4];

    uint32_t value;
    int32_t row_step;
    int32_t col_step;
    int32_t step;
    int32_t row_off;
    int32_t col_off;
    uint8_t row;
    uint8_t col;

    uint8_t up;
    uint8_t bottom;
    uint8_t left;
    uint8_t right;
    TMiscMaskMode mask_mode;
    bool invert;
};

STRUCT(TMiscMemInput) {
    uint8_t byte_idx;
    Tile data;
};

STRUCT(TMiscWriteback) {
    TOPRandType reg_type;
    Int<PTR_WID> pidx;
    uint8_t byte_idx;
    Tile data;
    bool last;
};
