#pragma once

#include <array>
#include <cassert>
#include <cstdint>

#include "../header/tmisc.hpp"

// Simulation-only Tensor construction and reshape unit. This module uses
// software-style state and loops for readability and simulation speed; it is not
// intended to be synthesizable.

REQUEST_READY(miscWriteback, ARG(TMiscWriteback) wb);

REGISTER(tmisc_sw_ready, bool) {
    tmisc_sw_ready = false;
}

HELPER() {
bool tmisc_busy;
TMiscInstMeta tmisc_inst;
Tile32 tmisc_src;
Tile32 tmisc_result;
std::array<bool, 4> tmisc_src_byte_valid;
bool tmisc_result_ready;
bool tmisc_wb_active;
uint8_t tmisc_wb_byte;

inline uint32_t tmisc_dtype_bytes(DType dtype) {
    uint32_t group = (static_cast<uint32_t>(dtype) >> 4U) & 0x3U;
    if (group == 0U) {
        return 1U;
    }
    if (group == 1U) {
        return 2U;
    }
    return 4U;
}

inline bool tmisc_is_signed(DType dtype) {
    return dtype == DType::DTYPE_SINT8 ||
           dtype == DType::DTYPE_SINT16 ||
           dtype == DType::DTYPE_SINT32;
}

inline uint32_t tmisc_dtype_mask(DType dtype) {
    uint32_t bytes = tmisc_dtype_bytes(dtype);
    if (bytes == 1U) {
        return 0xffU;
    }
    if (bytes == 2U) {
        return 0xffffU;
    }
    return 0xffffffffU;
}

inline bool tmisc_needs_input(TMiscOp op) {
    return op == TMiscOp::TBCAST ||
           op == TMiscOp::TBCAST_R ||
           op == TMiscOp::TBCAST_C ||
           op == TMiscOp::TSHIFT ||
           op == TMiscOp::TSHIFT_Z ||
           op == TMiscOp::TTRANS ||
           op == TMiscOp::TTRANS_RR ||
           op == TMiscOp::TTRANS_RC;
}

inline uint32_t tmisc_output_bytes(const TMiscInstMeta &inst) {
    if (inst.op == TMiscOp::TSETMASK) {
        return 1U;
    }
    return tmisc_dtype_bytes(inst.rd_datatype);
}

inline void tmisc_clear_tile32(Tile32 &tile) {
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            tile[r][c] = 0;
        }
    }
}

inline void tmisc_reset_sw() {
    tmisc_busy = false;
    tmisc_result_ready = false;
    tmisc_wb_active = false;
    tmisc_wb_byte = 0;
    tmisc_inst.op = TMiscOp::TFILL;
    tmisc_inst.rd_datatype = DType::DTYPE_UINT8;
    tmisc_inst.rs1_datatype = DType::DTYPE_UINT8;
    tmisc_inst.rd_type = TOPRandType::TREG;
    tmisc_inst.rs1_type = TOPRandType::TREG;
    tmisc_inst.value = 0;
    tmisc_inst.row_step = 0;
    tmisc_inst.col_step = 0;
    tmisc_inst.step = 0;
    tmisc_inst.row_off = 0;
    tmisc_inst.col_off = 0;
    tmisc_inst.row = 0;
    tmisc_inst.col = 0;
    tmisc_inst.up = 0;
    tmisc_inst.bottom = 0;
    tmisc_inst.left = 0;
    tmisc_inst.right = 0;
    tmisc_inst.mask_mode = TMiscMaskMode::RECT;
    tmisc_inst.invert = false;
    for (uint32_t i = 0; i < 4; ++i) {
        tmisc_inst.rd_pidx[i] = 0;
        tmisc_inst.rs1_pidx[i] = 0;
        tmisc_src_byte_valid[i] = false;
    }
    tmisc_clear_tile32(tmisc_src);
    tmisc_clear_tile32(tmisc_result);
}

inline bool tmisc_ready() {
    return !tmisc_busy;
}

inline bool tmisc_input_ready() {
    if (!tmisc_needs_input(tmisc_inst.op)) {
        return true;
    }
    uint32_t bytes = tmisc_dtype_bytes(tmisc_inst.rs1_datatype);
    for (uint32_t i = 0; i < bytes; ++i) {
        if (!tmisc_src_byte_valid[i]) {
            return false;
        }
    }
    return true;
}

inline uint32_t tmisc_pack_value(int64_t value, DType dtype) {
    return uint32_t(value) & tmisc_dtype_mask(dtype);
}

inline int64_t tmisc_signed_or_unsigned(uint32_t value, DType dtype) {
    uint32_t masked = value & tmisc_dtype_mask(dtype);
    if (!tmisc_is_signed(dtype)) {
        return int64_t(masked);
    }
    uint32_t bytes = tmisc_dtype_bytes(dtype);
    if (bytes == 1U) {
        int8_t v = int8_t(masked & 0xffU);
        return int64_t(v);
    }
    if (bytes == 2U) {
        int16_t v = int16_t(masked & 0xffffU);
        return int64_t(v);
    }
    int32_t v = int32_t(masked);
    return int64_t(v);
}

inline void tmisc_compute_fill(bool linear) {
    int64_t base = tmisc_signed_or_unsigned(tmisc_inst.value, tmisc_inst.rd_datatype);
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            int64_t v = base;
            if (linear) {
                v += int64_t(tmisc_inst.step) * int64_t(r * TILE_SIZE + c);
            } else {
                v += int64_t(tmisc_inst.row_step) * int64_t(r);
                v += int64_t(tmisc_inst.col_step) * int64_t(c);
            }
            tmisc_result[r][c] = tmisc_pack_value(v, tmisc_inst.rd_datatype);
        }
    }
}

inline void tmisc_compute_bcast() {
    uint32_t src_row = uint32_t(tmisc_inst.row) % TILE_SIZE;
    uint32_t src_col = uint32_t(tmisc_inst.col) % TILE_SIZE;
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            uint32_t rr = src_row;
            uint32_t cc = src_col;
            if (tmisc_inst.op == TMiscOp::TBCAST_R) {
                rr = r;
            } else if (tmisc_inst.op == TMiscOp::TBCAST_C) {
                cc = c;
            }
            tmisc_result[r][c] = tmisc_src[rr][cc] & tmisc_dtype_mask(tmisc_inst.rd_datatype);
        }
    }
}

inline int32_t tmisc_mod_index(int32_t value) {
    int32_t m = value % int32_t(TILE_SIZE);
    if (m < 0) {
        m += int32_t(TILE_SIZE);
    }
    return m;
}

inline void tmisc_compute_shift() {
    bool zero_fill = tmisc_inst.op == TMiscOp::TSHIFT_Z;
    for (int32_t r = 0; r < int32_t(TILE_SIZE); ++r) {
        for (int32_t c = 0; c < int32_t(TILE_SIZE); ++c) {
            int32_t rr = r + tmisc_inst.row_off;
            int32_t cc = c + tmisc_inst.col_off;
            if (zero_fill &&
                (rr < 0 || rr >= int32_t(TILE_SIZE) || cc < 0 || cc >= int32_t(TILE_SIZE))) {
                tmisc_result[uint32_t(r)][uint32_t(c)] = 0;
            } else {
                uint32_t sr = uint32_t(tmisc_mod_index(rr));
                uint32_t sc = uint32_t(tmisc_mod_index(cc));
                tmisc_result[uint32_t(r)][uint32_t(c)] =
                    tmisc_src[sr][sc] & tmisc_dtype_mask(tmisc_inst.rd_datatype);
            }
        }
    }
}

inline void tmisc_compute_trans() {
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            uint32_t rr = r;
            uint32_t cc = c;
            if (tmisc_inst.op == TMiscOp::TTRANS) {
                rr = c;
                cc = r;
            } else if (tmisc_inst.op == TMiscOp::TTRANS_RR) {
                rr = TILE_SIZE - 1U - r;
            } else if (tmisc_inst.op == TMiscOp::TTRANS_RC) {
                cc = TILE_SIZE - 1U - c;
            }
            tmisc_result[r][c] = tmisc_src[rr][cc] & tmisc_dtype_mask(tmisc_inst.rd_datatype);
        }
    }
}

inline bool tmisc_mask_rect_contains(uint32_t r, uint32_t c) {
    uint32_t top = tmisc_inst.up;
    uint32_t bottom = TILE_SIZE > uint32_t(tmisc_inst.bottom) ?
                      TILE_SIZE - uint32_t(tmisc_inst.bottom) : 0U;
    uint32_t left = tmisc_inst.left;
    uint32_t right = TILE_SIZE > uint32_t(tmisc_inst.right) ?
                     TILE_SIZE - uint32_t(tmisc_inst.right) : 0U;
    return r >= top && r < bottom && c >= left && c < right;
}

inline bool tmisc_mask_triangle_contains(uint32_t r, uint32_t c) {
    if (!tmisc_mask_rect_contains(r, c)) {
        return false;
    }
    int32_t lr = int32_t(r) - int32_t(tmisc_inst.up);
    int32_t lc = int32_t(c) - int32_t(tmisc_inst.left);
    if (tmisc_inst.mask_mode == TMiscMaskMode::UT) {
        return lc > lr;
    }
    if (tmisc_inst.mask_mode == TMiscMaskMode::UT_D) {
        return lc >= lr;
    }
    if (tmisc_inst.mask_mode == TMiscMaskMode::LT) {
        return lc < lr;
    }
    if (tmisc_inst.mask_mode == TMiscMaskMode::LT_D) {
        return lc <= lr;
    }
    return true;
}

inline void tmisc_compute_setmask() {
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            bool v = tmisc_mask_triangle_contains(r, c);
            if (tmisc_inst.invert) {
                v = !v;
            }
            tmisc_result[r][c] = v ? 1U : 0U;
        }
    }
}

inline void tmisc_compute_result() {
    tmisc_clear_tile32(tmisc_result);
    if (tmisc_inst.op == TMiscOp::TFILL) {
        tmisc_compute_fill(false);
    } else if (tmisc_inst.op == TMiscOp::TFILLLINE) {
        tmisc_compute_fill(true);
    } else if (tmisc_inst.op == TMiscOp::TBCAST ||
               tmisc_inst.op == TMiscOp::TBCAST_R ||
               tmisc_inst.op == TMiscOp::TBCAST_C) {
        tmisc_compute_bcast();
    } else if (tmisc_inst.op == TMiscOp::TSHIFT ||
               tmisc_inst.op == TMiscOp::TSHIFT_Z) {
        tmisc_compute_shift();
    } else if (tmisc_inst.op == TMiscOp::TTRANS ||
               tmisc_inst.op == TMiscOp::TTRANS_RR ||
               tmisc_inst.op == TMiscOp::TTRANS_RC) {
        tmisc_compute_trans();
    } else if (tmisc_inst.op == TMiscOp::TSETMASK) {
        tmisc_compute_setmask();
    }
    tmisc_result_ready = true;
    tmisc_wb_active = true;
    tmisc_wb_byte = 0;
}

inline Tile tmisc_make_wb_tile(uint32_t byte_idx) {
    Tile out;
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            out[r][c] = uint8_t((tmisc_result[r][c] >> (byte_idx * 8U)) & 0xffU);
        }
    }
    return out;
}

}

SERVICE_READY(s0MiscFire, tmisc_ready(), ARG(TMiscInstMeta) inst) {
    tmisc_busy = true;
    tmisc_inst = inst;
    tmisc_clear_tile32(tmisc_src);
    tmisc_clear_tile32(tmisc_result);
    for (uint32_t i = 0; i < 4; ++i) {
        tmisc_src_byte_valid[i] = false;
    }
    tmisc_result_ready = false;
    tmisc_wb_active = false;
    tmisc_wb_byte = 0;
    if (!tmisc_needs_input(inst.op)) {
        tmisc_compute_result();
    }
}

SERVICE(miscMemInput, ARG(TMiscMemInput) in) {
    if (!tmisc_busy || !tmisc_needs_input(tmisc_inst.op) || in.byte_idx >= 4U) {
        return;
    }
    uint32_t bytes = tmisc_dtype_bytes(tmisc_inst.rs1_datatype);
    if (in.byte_idx >= bytes) {
        return;
    }
    uint32_t shift = uint32_t(in.byte_idx) * 8U;
    uint32_t clear_mask = ~(0xffU << shift);
    for (uint32_t r = 0; r < TILE_SIZE; ++r) {
        for (uint32_t c = 0; c < TILE_SIZE; ++c) {
            uint32_t cur = tmisc_src[r][c] & clear_mask;
            tmisc_src[r][c] = cur | (uint32_t(in.data[r][c]) << shift);
        }
    }
    tmisc_src_byte_valid[in.byte_idx] = true;
    if (tmisc_input_ready() && !tmisc_result_ready) {
        tmisc_compute_result();
    }
}

TICK_IMPL() {
    if (!tmisc_sw_ready) {
        tmisc_reset_sw();
        tmisc_sw_ready.setnext(true);
    }

    if (tmisc_wb_active) {
        TMiscWriteback wb;
        wb.reg_type = tmisc_inst.rd_type;
        wb.pidx = tmisc_inst.rd_pidx[tmisc_wb_byte];
        wb.byte_idx = tmisc_wb_byte;
        wb.data = tmisc_make_wb_tile(tmisc_wb_byte);
        wb.last = uint32_t(tmisc_wb_byte) + 1U >= tmisc_output_bytes(tmisc_inst);
        if (miscWriteback(wb)) {
            if (wb.last) {
                tmisc_busy = false;
                tmisc_result_ready = false;
                tmisc_wb_active = false;
                tmisc_wb_byte = 0;
            } else {
                tmisc_wb_byte++;
            }
        }
    }
}
