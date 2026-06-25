#pragma once

#include "inst.hpp"
#include "tile.hpp"

STRUCT(AluInstMeta) {
    AluOp op;
    DType rd_datatype;
    DType rs1_datatype;
    DType rs2_datatype;
    TOPRandType rd_type;
    TOPRandType rs1_type;
    TOPRandType rs2_type;
    Int<PTR_WID> rd_pidx[4];
    Int<PTR_WID> rs1_pidx[4];
    Int<PTR_WID> rs2_pidx[4];
    Int<PMASK_WID> pmask;
    bool mask_enable;
};

STRUCT(AluMemInput) {
    uint8_t operand;
    uint8_t byte_idx;
    Tile data;
};

STRUCT(AluWriteback) {
    TOPRandType reg_type;
    Int<PTR_WID> pidx;
    uint8_t byte_idx;
    Tile data;
    bool last;
};

STRUCT(AluAccgetMeta) {
    uint8_t pacc;
    uint8_t ptemp_rs;
    uint8_t ptemp_rd;
    DType dtype;
    Int<5> funct5;
};

STRUCT(AluRetireInfo) {
    TOPRandType reg_type;
    Int<PTR_WID> pidx[4];
    Int<4> valid;
};
