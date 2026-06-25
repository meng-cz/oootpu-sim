#pragma once

#include "defhelper.hpp"

ENUM(SubOPCode) {
    // 暂时先放GEMM相关的指令，其他指令后续再加
    GEMM = 0x70,
    TACCGET = 0x71,
};

ENUM(AluOp) {
    TADD = 0x00,
    TSUB = 0x01,
    TMUL = 0x02,
    TDIV = 0x03,
    TMAX = 0x04,
    TMIN = 0x05,
    TAND = 0x06,
    TOR = 0x07,
    TXOR = 0x08,
    TSLL = 0x09,
    TSRL = 0x0a,
    TSRA = 0x0b,
    TMERGE = 0x0c,
    TNOT = 0x10,
    TABS = 0x11,
    TNEG = 0x12,
    TSQRT = 0x13,
    TRSQRT = 0x14,
    TEXP = 0x15,
    TRCP = 0x16,
    TRELU = 0x17,
    TRELU6 = 0x18,
    TSIGMOID = 0x19,
    TTANH = 0x1a,
    TGELU = 0x1b,
    TCVT = 0x1c,
    TMMV = 0x20,
    TMNOT = 0x21,
    TMAND = 0x22,
    TMNAND = 0x23,
    TMOR = 0x24,
    TMNOR = 0x25,
    TMXOR = 0x26,
    TMNXOR = 0x27,
    TMEQ = 0x28,
    TMNEQ = 0x29,
    TMLT = 0x2a,
    TMGE = 0x2b,
    TMLE = 0x2c,
    TMGT = 0x2d
};
