#pragma once

#include "defhelper.hpp"

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
