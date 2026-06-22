#pragma once

#include <cstdint>
#include <defhelper.hpp>

CONFIG(DEPTH_A, 9);
CONFIG(DEPTH_B, 16);
CONFIG(WIDTH_A, clog2(DEPTH_A));
CONFIG(WIDTH_B, @DEPTH_B);
CONFIG(WIDTH_SUM, clog2(DEPTH_A + DEPTH_B));

ALIAS(IdxA, Int<WIDTH_A>);
ALIAS(IdxB, Int<WIDTH_B>);

STRUCT(Clog2Status) {
    uint32_t width_a;
    uint32_t width_b;
    uint32_t width_sum;
    bool a_wraps;
    bool b_wraps;
    IdxA idx_a;
    IdxB idx_b;
};
