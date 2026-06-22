#pragma once

#include "header.hpp"

REGISTER(idx_a, IdxA) {
    idx_a = 0;
}

REGISTER(idx_b, IdxB) {
    idx_b = 0;
}

SERVICE(step) {
    uint32_t next_a = (idx_a.get().to<uint32_t>() + 1U) % DEPTH_A;
    uint32_t next_b = (idx_b.get().to<uint32_t>() + 1U) % DEPTH_B;
    idx_a.setnext(static_cast<IdxA>(next_a));
    idx_b.setnext(static_cast<IdxB>(next_b));
}

QUERY(status, Clog2Status) {
    Clog2Status s;
    s.width_a = WIDTH_A;
    s.width_b = WIDTH_B;
    s.width_sum = WIDTH_SUM;
    s.a_wraps = (idx_a.get() == static_cast<IdxA>(DEPTH_A - 1));
    s.b_wraps = (idx_b.get() == static_cast<IdxB>(DEPTH_B - 1));
    s.idx_a = idx_a.get();
    s.idx_b = idx_b.get();
    return s;
}
