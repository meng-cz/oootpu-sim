#pragma once

#include "config.hpp"
#include "dtype.hpp"
#include "opcode.hpp"
#include "operand.hpp"

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
