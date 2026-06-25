#pragma once

#include "inst.hpp"

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
