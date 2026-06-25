#pragma once

#include <array>
#include <cassert>
#include <cstdint>

#include "header.hpp"

// Simulation-only SIMD Tile ALU. This model favors readable software flow over
// synthesizable hardware structure.

CONFIG(ALU_COPY_LATENCY, 0);
CONFIG(ALU_INT_SIMPLE_LATENCY, 1);
CONFIG(ALU_INT_MUL_LATENCY, 3);
CONFIG(ALU_FLOAT_BASIC_LATENCY, 4);
CONFIG(ALU_FLOAT_SQRT_LATENCY, 12);
CONFIG(ALU_FLOAT_RSQRT_LATENCY, 14);
CONFIG(ALU_FLOAT_EXP_LATENCY, 16);
CONFIG(ALU_FLOAT_RCP_LATENCY, 12);
CONFIG(ALU_FLOAT_SIGMOID_LATENCY, 16);
CONFIG(ALU_FLOAT_TANH_LATENCY, 16);
CONFIG(ALU_FLOAT_GELU_LATENCY, 16);

REQUEST_READY(aluWriteback, ARG(AluWriteback) wb);
REQUEST_READY(s1FireAccget, ARG(uint8_t) pacc, ARG(uint8_t) ptemp, ARG(DType) dtype);
REQUEST(sNAccOutput, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data);
REQUEST_READY(retireInst, ARG(AluRetireInfo) info);

REGISTER(alu_sw_ready, bool) {
    alu_sw_ready = false;
}

HELPER() {
std::array<Tile32, PTEMP_NUM> alu_temps;

bool alu_busy;
AluInstMeta alu_inst;
std::array<Tile32, 2> alu_src;
Tile32 alu_result;
std::array<std::array<bool, 4>, 2> alu_src_byte_valid;
std::array<bool, 2> alu_src_ready;
Tile alu_mask;
bool alu_mask_ready;
bool alu_result_ready;
bool alu_exec_active;
uint32_t alu_exec_remain;
bool alu_wb_active;
uint8_t alu_wb_byte;
AluRetireInfo alu_retire_info;
bool alu_retire_pending;

bool acc_busy;
bool acc_started;
uint8_t acc_send_lane;
AluAccgetMeta acc_meta;
bool acc_retire_pending;
AluRetireInfo acc_retire_info;

inline uint32_t alu_dtype_bytes(DType dtype) {
    uint32_t group = (static_cast<uint32_t>(dtype) >> 4U) & 0x3U;
    if (group == 0U) {
        return 1U;
    }
    if (group == 1U) {
        return 2U;
    }
    return 4U;
}

inline bool alu_is_signed(DType dtype) {
    return dtype == DType::DTYPE_SINT8 ||
           dtype == DType::DTYPE_SINT16 ||
           dtype == DType::DTYPE_SINT32;
}

inline bool alu_is_unsigned(DType dtype) {
    return dtype == DType::DTYPE_UINT8 ||
           dtype == DType::DTYPE_UINT16 ||
           dtype == DType::DTYPE_UINT32;
}

inline uint32_t alu_mask_value(uint32_t value, DType dtype) {
    uint32_t bytes = alu_dtype_bytes(dtype);
    if (bytes == 1U) {
        return value & 0xffU;
    }
    if (bytes == 2U) {
        return value & 0xffffU;
    }
    return value;
}

inline float alu_u32_to_f32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

inline uint32_t alu_f32_to_u32(float f) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = f;
    return v.u;
}

inline bool alu_is_float_dtype(DType dtype) {
    return dtype == DType::DTYPE_F8E4M3 ||
           dtype == DType::DTYPE_F8E5M2 ||
           dtype == DType::DTYPE_F16 ||
           dtype == DType::DTYPE_F32;
}

inline bool alu_same_dtype_class(DType lhs, DType rhs) {
    return (alu_is_signed(lhs) && alu_is_signed(rhs)) ||
           (alu_is_unsigned(lhs) && alu_is_unsigned(rhs)) ||
           (alu_is_float_dtype(lhs) && alu_is_float_dtype(rhs));
}

inline DType alu_mul_output_dtype(DType src_dtype, DType rd_dtype) {
    if (alu_same_dtype_class(src_dtype, rd_dtype) &&
        alu_dtype_bytes(rd_dtype) > alu_dtype_bytes(src_dtype)) {
        return rd_dtype;
    }
    return src_dtype;
}

inline float alu_pow2_int(int exp) {
    float value = 1.0f;
    if (exp >= 0) {
        for (int i = 0; i < exp; ++i) {
            value *= 2.0f;
        }
    } else {
        for (int i = 0; i < -exp; ++i) {
            value *= 0.5f;
        }
    }
    return value;
}

inline float alu_decode_custom_float(uint32_t bits, uint32_t exp_bits, uint32_t mant_bits, int bias) {
    uint32_t sign = bits >> (exp_bits + mant_bits);
    uint32_t exp_mask = (1U << exp_bits) - 1U;
    uint32_t mant_mask = (1U << mant_bits) - 1U;
    uint32_t exp = (bits >> mant_bits) & exp_mask;
    uint32_t mant = bits & mant_mask;
    float mag = 0.0f;
    if (exp == 0U) {
        if (mant == 0U) {
            mag = 0.0f;
        } else {
            mag = (float(mant) / float(1U << mant_bits)) * alu_pow2_int(1 - bias);
        }
    } else if (exp == exp_mask) {
        mag = 3.402823e38f;
    } else {
        mag = (1.0f + float(mant) / float(1U << mant_bits)) * alu_pow2_int(int(exp) - bias);
    }
    return sign != 0U ? -mag : mag;
}

inline uint32_t alu_encode_custom_float(float value, uint32_t exp_bits, uint32_t mant_bits, int bias) {
    uint32_t sign = value < 0.0f ? 1U : 0U;
    float mag = value < 0.0f ? -value : value;
    uint32_t exp_mask = (1U << exp_bits) - 1U;
    if (mag == 0.0f) {
        return sign << (exp_bits + mant_bits);
    }

    int exp = 0;
    float norm = mag;
    while (norm >= 2.0f) {
        norm *= 0.5f;
        exp++;
    }
    while (norm < 1.0f) {
        norm *= 2.0f;
        exp--;
    }

    int encoded_exp = exp + bias;
    if (encoded_exp >= int(exp_mask)) {
        return (sign << (exp_bits + mant_bits)) | (exp_mask << mant_bits);
    }
    if (encoded_exp <= 0) {
        return sign << (exp_bits + mant_bits);
    }

    float mant_f = (norm - 1.0f) * float(1U << mant_bits);
    uint32_t mant = uint32_t(mant_f + 0.5f);
    if (mant >= (1U << mant_bits)) {
        mant = 0;
        encoded_exp++;
        if (encoded_exp >= int(exp_mask)) {
            return (sign << (exp_bits + mant_bits)) | (exp_mask << mant_bits);
        }
    }
    return (sign << (exp_bits + mant_bits)) | (uint32_t(encoded_exp) << mant_bits) | mant;
}

inline float alu_decode_float(uint32_t bits, DType dtype) {
    if (dtype == DType::DTYPE_F32) {
        return alu_u32_to_f32(bits);
    }
    if (dtype == DType::DTYPE_F16) {
        return alu_decode_custom_float(bits & 0xffffU, 5, 10, 15);
    }
    if (dtype == DType::DTYPE_F8E5M2) {
        return alu_decode_custom_float(bits & 0xffU, 5, 2, 15);
    }
    return alu_decode_custom_float(bits & 0xffU, 4, 3, 7);
}

inline uint32_t alu_encode_float(float value, DType dtype) {
    if (dtype == DType::DTYPE_F32) {
        return alu_f32_to_u32(value);
    }
    if (dtype == DType::DTYPE_F16) {
        return alu_encode_custom_float(value, 5, 10, 15);
    }
    if (dtype == DType::DTYPE_F8E5M2) {
        return alu_encode_custom_float(value, 5, 2, 15);
    }
    return alu_encode_custom_float(value, 4, 3, 7);
}

inline int64_t alu_decode_int(uint32_t value, DType dtype) {
    if (dtype == DType::DTYPE_SINT8) {
        return int64_t(static_cast<int8_t>(value & 0xffU));
    }
    if (dtype == DType::DTYPE_SINT16) {
        return int64_t(static_cast<int16_t>(value & 0xffffU));
    }
    if (dtype == DType::DTYPE_SINT32) {
        return int64_t(static_cast<int32_t>(value));
    }
    return int64_t(alu_mask_value(value, dtype));
}

inline float alu_absf(float x) {
    return x < 0.0f ? -x : x;
}

inline float alu_sqrt_approx(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    float y = x > 1.0f ? x : 1.0f;
    for (int i = 0; i < 8; ++i) {
        y = 0.5f * (y + x / y);
    }
    return y;
}

inline float alu_exp_approx(float x) {
    if (x < -16.0f) {
        return 0.0f;
    }
    if (x > 16.0f) {
        x = 16.0f;
    }
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 12; ++i) {
        term = term * x / float(i);
        sum += term;
    }
    return sum < 0.0f ? 0.0f : sum;
}

inline float alu_tanh_approx(float x) {
    float e2 = alu_exp_approx(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

inline bool alu_is_binary(AluOp op) {
    return op == AluOp::TADD || op == AluOp::TSUB || op == AluOp::TMUL ||
           op == AluOp::TDIV || op == AluOp::TMAX || op == AluOp::TMIN ||
           op == AluOp::TAND || op == AluOp::TOR || op == AluOp::TXOR ||
           op == AluOp::TSLL || op == AluOp::TSRL || op == AluOp::TSRA ||
           op == AluOp::TMERGE || op == AluOp::TMAND || op == AluOp::TMNAND ||
           op == AluOp::TMOR || op == AluOp::TMNOR || op == AluOp::TMXOR ||
           op == AluOp::TMNXOR || op == AluOp::TMEQ || op == AluOp::TMNEQ ||
           op == AluOp::TMLT || op == AluOp::TMGE || op == AluOp::TMLE ||
           op == AluOp::TMGT;
}

inline bool alu_is_mask_op(AluOp op) {
    return op == AluOp::TMMV || op == AluOp::TMNOT ||
           op == AluOp::TMAND || op == AluOp::TMNAND ||
           op == AluOp::TMOR || op == AluOp::TMNOR ||
           op == AluOp::TMXOR || op == AluOp::TMNXOR ||
           op == AluOp::TMEQ || op == AluOp::TMNEQ ||
           op == AluOp::TMLT || op == AluOp::TMGE ||
           op == AluOp::TMLE || op == AluOp::TMGT;
}

inline bool alu_needs_src2(AluOp op) {
    return alu_is_binary(op);
}

inline bool alu_needs_mask(AluOp op, bool mask_enable) {
    return mask_enable || op == AluOp::TMERGE;
}

inline uint32_t alu_simd_chunks() {
    return (TILE_SIZE + SIMD_LANE_SIZE - 1U) / SIMD_LANE_SIZE;
}

inline bool alu_is_compare_op(AluOp op) {
    return op == AluOp::TMEQ || op == AluOp::TMNEQ ||
           op == AluOp::TMLT || op == AluOp::TMGE ||
           op == AluOp::TMLE || op == AluOp::TMGT;
}

inline bool alu_is_temp_copy_inst() {
    return alu_inst.op == AluOp::TADD &&
           alu_inst.rs2_type == TOPRandType::INVALID &&
           (alu_inst.rd_type == TOPRandType::TTEMP ||
            alu_inst.rs1_type == TOPRandType::TTEMP);
}

inline bool alu_is_float_op() {
    return alu_is_float_dtype(alu_inst.rs1_datatype) ||
           (alu_inst.op == AluOp::TMUL && alu_is_float_dtype(alu_inst.rd_datatype));
}

inline uint32_t alu_op_pipe_latency() {
    if (alu_is_temp_copy_inst()) {
        return ALU_COPY_LATENCY;
    }
    if (!alu_is_float_op()) {
        return alu_inst.op == AluOp::TMUL ? ALU_INT_MUL_LATENCY : ALU_INT_SIMPLE_LATENCY;
    }
    if (alu_inst.op == AluOp::TSQRT) return ALU_FLOAT_SQRT_LATENCY;
    if (alu_inst.op == AluOp::TRSQRT) return ALU_FLOAT_RSQRT_LATENCY;
    if (alu_inst.op == AluOp::TEXP) return ALU_FLOAT_EXP_LATENCY;
    if (alu_inst.op == AluOp::TRCP) return ALU_FLOAT_RCP_LATENCY;
    if (alu_inst.op == AluOp::TSIGMOID) return ALU_FLOAT_SIGMOID_LATENCY;
    if (alu_inst.op == AluOp::TTANH) return ALU_FLOAT_TANH_LATENCY;
    if (alu_inst.op == AluOp::TGELU) return ALU_FLOAT_GELU_LATENCY;
    return ALU_FLOAT_BASIC_LATENCY;
}

inline uint32_t alu_total_exec_latency() {
    uint32_t pipe = alu_op_pipe_latency();
    if (pipe == 0U) {
        return 0U;
    }
    return pipe + alu_simd_chunks() - 1U;
}

inline void alu_clear_tile32(Tile32 &tile) {
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            tile[r][c] = 0;
        }
    }
}

inline void alu_reset_sw() {
    alu_busy = false;
    alu_result_ready = false;
    alu_exec_active = false;
    alu_exec_remain = 0;
    alu_wb_active = false;
    alu_wb_byte = 0;
    alu_retire_pending = false;
    alu_mask_ready = false;
    for (int t = 0; t < PTEMP_NUM; ++t) {
        alu_clear_tile32(alu_temps[t]);
    }
    for (int s = 0; s < 2; ++s) {
        alu_src_ready[s] = false;
        for (int b = 0; b < 4; ++b) {
            alu_src_byte_valid[s][b] = false;
        }
        alu_clear_tile32(alu_src[s]);
    }
    alu_clear_tile32(alu_result);
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            alu_mask[r][c] = 0;
        }
    }
    alu_retire_info.reg_type = TOPRandType::INVALID;
    for (int i = 0; i < 4; ++i) {
        alu_retire_info.pidx[i] = 0;
    }
    alu_retire_info.valid = 0;

    acc_busy = false;
    acc_started = false;
    acc_send_lane = 0;
    acc_retire_pending = false;
    acc_retire_info.reg_type = TOPRandType::INVALID;
    for (int i = 0; i < 4; ++i) {
        acc_retire_info.pidx[i] = 0;
    }
    acc_retire_info.valid = 0;
}

inline bool alu_ready_for_inst() {
    return !alu_busy && !alu_exec_active && !alu_wb_active && !alu_retire_pending;
}

inline bool alu_accget_ready() {
    return !acc_busy && !acc_retire_pending;
}

inline void alu_init_operand_from_temp(uint32_t src_slot, uint32_t ptemp) {
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            alu_src[src_slot][r][c] = alu_temps[ptemp][r][c];
        }
    }
    alu_src_ready[src_slot] = true;
}

inline bool alu_operand_ready(uint32_t src_slot, TOPRandType type, DType dtype) {
    if (type == TOPRandType::INVALID || type == TOPRandType::TACC) {
        return true;
    }
    if (type == TOPRandType::TTEMP) {
        return true;
    }
    if (type == TOPRandType::TMASK) {
        return alu_src_byte_valid[src_slot][0];
    }
    uint32_t bytes = alu_dtype_bytes(dtype);
    for (uint32_t b = 0; b < bytes; ++b) {
        if (!alu_src_byte_valid[src_slot][b]) {
            return false;
        }
    }
    return true;
}

inline bool alu_inputs_ready() {
    if (!alu_operand_ready(0, alu_inst.rs1_type, alu_inst.rs1_datatype)) {
        return false;
    }
    if (alu_needs_src2(alu_inst.op) &&
        !alu_operand_ready(1, alu_inst.rs2_type, alu_inst.rs2_datatype)) {
        return false;
    }
    if (alu_needs_mask(alu_inst.op, alu_inst.mask_enable) && !alu_mask_ready) {
        return false;
    }
    return true;
}

inline bool alu_bool_at(const Tile32 &tile, int r, int c) {
    return (tile[r][c] & 0xffU) != 0U;
}

inline uint32_t alu_binary_value(AluOp op, uint32_t a, uint32_t b, DType src_dtype, DType rd_dtype) {
    DType out_dtype = op == AluOp::TMUL ? alu_mul_output_dtype(src_dtype, rd_dtype) : src_dtype;
    if (alu_is_float_dtype(src_dtype)) {
        float fa = alu_decode_float(a, src_dtype);
        float fb = alu_decode_float(b, src_dtype);
        float out = 0.0f;
        if (op == AluOp::TADD) out = fa + fb;
        else if (op == AluOp::TSUB) out = fa - fb;
        else if (op == AluOp::TMUL) out = fa * fb;
        else if (op == AluOp::TDIV) out = 0.0f;
        else if (op == AluOp::TMAX) out = fa > fb ? fa : fb;
        else if (op == AluOp::TMIN) out = fa < fb ? fa : fb;
        return alu_encode_float(out, out_dtype);
    }

    uint32_t mask = alu_dtype_bytes(out_dtype) == 1U ? 0xffU :
                    (alu_dtype_bytes(out_dtype) == 2U ? 0xffffU : 0xffffffffU);
    uint32_t shamt = b & 31U;
    uint64_t ua = uint64_t(alu_mask_value(a, src_dtype));
    uint64_t ub = uint64_t(alu_mask_value(b, src_dtype));
    int64_t sa = alu_decode_int(a, src_dtype);
    int64_t sb = alu_decode_int(b, src_dtype);
    int64_t out = 0;
    bool signed_op = alu_is_signed(src_dtype);
    if (op == AluOp::TADD) out = signed_op ? sa + sb : int64_t(ua + ub);
    else if (op == AluOp::TSUB) out = signed_op ? sa - sb : int64_t(ua - ub);
    else if (op == AluOp::TMUL) out = signed_op ? sa * sb : int64_t(ua * ub);
    else if (op == AluOp::TDIV) out = 0;
    else if (op == AluOp::TMAX) out = signed_op ? (sa > sb ? sa : sb) : int64_t(ua > ub ? ua : ub);
    else if (op == AluOp::TMIN) out = signed_op ? (sa < sb ? sa : sb) : int64_t(ua < ub ? ua : ub);
    else if (op == AluOp::TAND) out = int64_t(ua & ub);
    else if (op == AluOp::TOR) out = int64_t(ua | ub);
    else if (op == AluOp::TXOR) out = int64_t(ua ^ ub);
    else if (op == AluOp::TSLL) out = int64_t(ua << shamt);
    else if (op == AluOp::TSRL) out = int64_t(ua >> shamt);
    else if (op == AluOp::TSRA) out = sa >> shamt;
    return uint32_t(out) & mask;
}

inline uint32_t alu_unary_value(AluOp op, uint32_t a, DType dtype) {
    if (alu_is_float_dtype(dtype)) {
        float fa = alu_decode_float(a, dtype);
        float out = fa;
        if (op == AluOp::TABS) out = alu_absf(fa);
        else if (op == AluOp::TNEG) out = -fa;
        else if (op == AluOp::TSQRT) out = alu_sqrt_approx(fa);
        else if (op == AluOp::TRSQRT) out = 1.0f / alu_sqrt_approx(fa);
        else if (op == AluOp::TEXP) out = alu_exp_approx(fa);
        else if (op == AluOp::TRCP) out = 1.0f / fa;
        else if (op == AluOp::TRELU) out = fa > 0.0f ? fa : 0.0f;
        else if (op == AluOp::TRELU6) out = fa < 0.0f ? 0.0f : (fa > 6.0f ? 6.0f : fa);
        else if (op == AluOp::TSIGMOID) out = 1.0f / (1.0f + alu_exp_approx(-fa));
        else if (op == AluOp::TTANH) out = alu_tanh_approx(fa);
        else if (op == AluOp::TGELU) out = 0.5f * fa * (1.0f + alu_tanh_approx(0.79788456f * (fa + 0.044715f * fa * fa * fa)));
        return alu_encode_float(out, dtype);
    }

    uint32_t mask = alu_dtype_bytes(dtype) == 1U ? 0xffU :
                    (alu_dtype_bytes(dtype) == 2U ? 0xffffU : 0xffffffffU);
    int64_t sa = alu_decode_int(a, dtype);
    uint64_t ua = uint64_t(alu_mask_value(a, dtype));
    int64_t out = int64_t(ua);
    if (op == AluOp::TNOT) out = int64_t(~ua);
    else if (op == AluOp::TABS) out = sa < 0 ? -sa : sa;
    else if (op == AluOp::TNEG) out = -sa;
    else if (op == AluOp::TRELU) out = sa > 0 ? sa : 0;
    else if (op == AluOp::TRELU6) out = sa < 0 ? 0 : (sa > 6 ? 6 : sa);
    else if (op == AluOp::TCVT) out = sa;
    return uint32_t(out) & mask;
}

inline bool alu_compare_value(AluOp op, uint32_t a, uint32_t b, DType dtype) {
    if (alu_is_float_dtype(dtype)) {
        float fa = alu_decode_float(a, dtype);
        float fb = alu_decode_float(b, dtype);
        if (op == AluOp::TMEQ) return fa == fb;
        if (op == AluOp::TMNEQ) return fa != fb;
        if (op == AluOp::TMLT) return fa < fb;
        if (op == AluOp::TMGE) return fa >= fb;
        if (op == AluOp::TMLE) return fa <= fb;
        if (op == AluOp::TMGT) return fa > fb;
    }
    int64_t sa = alu_decode_int(a, dtype);
    int64_t sb = alu_decode_int(b, dtype);
    uint32_t ua = alu_mask_value(a, dtype);
    uint32_t ub = alu_mask_value(b, dtype);
    bool signed_cmp = alu_is_signed(dtype);
    if (op == AluOp::TMEQ) return ua == ub;
    if (op == AluOp::TMNEQ) return ua != ub;
    if (op == AluOp::TMLT) return signed_cmp ? sa < sb : ua < ub;
    if (op == AluOp::TMGE) return signed_cmp ? sa >= sb : ua >= ub;
    if (op == AluOp::TMLE) return signed_cmp ? sa <= sb : ua <= ub;
    if (op == AluOp::TMGT) return signed_cmp ? sa > sb : ua > ub;
    return false;
}

inline void alu_execute_current() {
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            bool write_ok = !alu_inst.mask_enable || alu_mask[r][c] != 0U;
            uint32_t a = alu_src[0][r][c];
            uint32_t b = alu_src[1][r][c];
            uint32_t out = 0;

            if (!write_ok) {
                out = 0;
            } else if (alu_inst.op == AluOp::TMERGE) {
                out = alu_mask[r][c] != 0U ? a : b;
            } else if (alu_inst.op == AluOp::TMMV) {
                out = alu_bool_at(alu_src[0], r, c) ? 1U : 0U;
            } else if (alu_inst.op == AluOp::TMNOT) {
                out = alu_bool_at(alu_src[0], r, c) ? 0U : 1U;
            } else if (alu_inst.op == AluOp::TMAND || alu_inst.op == AluOp::TMNAND ||
                       alu_inst.op == AluOp::TMOR || alu_inst.op == AluOp::TMNOR ||
                       alu_inst.op == AluOp::TMXOR || alu_inst.op == AluOp::TMNXOR) {
                bool ba = alu_bool_at(alu_src[0], r, c);
                bool bb = alu_bool_at(alu_src[1], r, c);
                bool bo = false;
                if (alu_inst.op == AluOp::TMAND) bo = ba && bb;
                else if (alu_inst.op == AluOp::TMNAND) bo = !(ba && bb);
                else if (alu_inst.op == AluOp::TMOR) bo = ba || bb;
                else if (alu_inst.op == AluOp::TMNOR) bo = !(ba || bb);
                else if (alu_inst.op == AluOp::TMXOR) bo = ba != bb;
                else bo = ba == bb;
                out = bo ? 1U : 0U;
            } else if (alu_inst.op == AluOp::TMEQ || alu_inst.op == AluOp::TMNEQ ||
                       alu_inst.op == AluOp::TMLT || alu_inst.op == AluOp::TMGE ||
                       alu_inst.op == AluOp::TMLE || alu_inst.op == AluOp::TMGT) {
                out = alu_compare_value(alu_inst.op, a, b, alu_inst.rs1_datatype) ? 1U : 0U;
            } else if (alu_is_binary(alu_inst.op)) {
                out = alu_binary_value(alu_inst.op, a, b, alu_inst.rs1_datatype, alu_inst.rd_datatype);
            } else {
                out = alu_unary_value(alu_inst.op, a, alu_inst.rs1_datatype);
            }
            alu_result[r][c] = out;
        }
    }
}

inline void alu_finish_execute() {
    alu_result_ready = true;
    alu_exec_active = false;
    alu_exec_remain = 0;
    if (alu_inst.rd_type == TOPRandType::TTEMP) {
        uint32_t ptemp = alu_inst.rd_pidx[0].template to<uint32_t>();
        if (ptemp < PTEMP_NUM) {
            alu_temps[ptemp] = alu_result;
        }
        alu_retire_info.reg_type = TOPRandType::TTEMP;
        for (int i = 0; i < 4; ++i) {
            alu_retire_info.pidx[i] = 0;
        }
        alu_retire_info.pidx[0] = ptemp;
        alu_retire_info.valid = 0x1;
        alu_retire_pending = true;
    } else if (alu_inst.rd_type == TOPRandType::TREG || alu_inst.rd_type == TOPRandType::TMASK) {
        alu_wb_active = true;
        alu_wb_byte = 0;
    } else {
        alu_retire_info.reg_type = TOPRandType::INVALID;
        for (int i = 0; i < 4; ++i) {
            alu_retire_info.pidx[i] = 0;
        }
        alu_retire_info.valid = 0;
        alu_retire_pending = true;
    }
}

inline void alu_start_inst(const AluInstMeta &inst) {
    alu_inst = inst;
    alu_busy = true;
    alu_result_ready = false;
    alu_exec_active = false;
    alu_exec_remain = 0;
    alu_wb_active = false;
    alu_wb_byte = 0;
    alu_retire_pending = false;
    alu_mask_ready = !alu_needs_mask(inst.op, inst.mask_enable);
    alu_clear_tile32(alu_result);
    for (int s = 0; s < 2; ++s) {
        alu_clear_tile32(alu_src[s]);
        alu_src_ready[s] = false;
        for (int b = 0; b < 4; ++b) {
            alu_src_byte_valid[s][b] = false;
        }
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            alu_mask[r][c] = 0;
        }
    }
    if (inst.rs1_type == TOPRandType::TTEMP) {
        alu_init_operand_from_temp(0, inst.rs1_pidx[0].template to<uint32_t>());
    } else if (inst.rs1_type == TOPRandType::INVALID) {
        alu_src_ready[0] = true;
    }
    if (inst.rs2_type == TOPRandType::TTEMP) {
        alu_init_operand_from_temp(1, inst.rs2_pidx[0].template to<uint32_t>());
    } else if (inst.rs2_type == TOPRandType::INVALID) {
        alu_src_ready[1] = true;
    }
}
}

SERVICE_READY(s0AluInstFire, alu_ready_for_inst(), ARG(AluInstMeta) inst) {
    alu_start_inst(inst);
}

SERVICE(sNAluMemInput, ARG(AluMemInput) in) {
    if (!alu_busy) {
        return;
    }
    if (in.operand == 3U) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                alu_mask[r][c] = in.data[r][c] != 0U ? 1U : 0U;
            }
        }
        alu_mask_ready = true;
        return;
    }
    if (in.operand < 1U || in.operand > 2U || in.byte_idx >= 4U) {
        return;
    }
    uint32_t src = in.operand - 1U;
    uint32_t shift = uint32_t(in.byte_idx) * 8U;
    uint32_t clear_mask = ~(0xffU << shift);
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            uint32_t cur = alu_src[src][r][c] & clear_mask;
            alu_src[src][r][c] = cur | (uint32_t(in.data[r][c]) << shift);
        }
    }
    alu_src_byte_valid[src][in.byte_idx] = true;
}

SERVICE_READY(s0AccgetStart, alu_accget_ready(), ARG(AluAccgetMeta) meta) {
    acc_meta = meta;
    acc_busy = true;
    acc_started = false;
    acc_send_lane = 0;
}

SERVICE(accTempInput, ARG(uint8_t) ptemp, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data) {
    uint32_t base_col = uint32_t(simdlane) * SIMD_LANE_SIZE;
    if (ptemp >= PTEMP_NUM || base_col >= TILE_SIZE) {
        return;
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int lane = 0; lane < SIMD_LANE_SIZE; ++lane) {
            alu_temps[ptemp][r][base_col + uint32_t(lane)] = data[r][lane];
        }
    }
}

SERVICE(accgetRetirePTemp, ARG(uint8_t) ptemp) {
    acc_retire_info.reg_type = TOPRandType::TTEMP;
    for (int i = 0; i < 4; ++i) {
        acc_retire_info.pidx[i] = 0;
    }
    acc_retire_info.pidx[0] = ptemp;
    acc_retire_info.valid = 0x1;
    acc_retire_pending = true;
}

TICK_IMPL() {
    if (!alu_sw_ready) {
        alu_reset_sw();
        alu_sw_ready.setnext(true);
    }

    bool exec_started = false;
    if (alu_busy && !alu_result_ready && !alu_exec_active && alu_inputs_ready()) {
        alu_execute_current();
        alu_exec_remain = alu_total_exec_latency();
        if (alu_exec_remain == 0U) {
            alu_finish_execute();
        } else {
            alu_exec_active = true;
            exec_started = true;
        }
    }

    if (alu_exec_active && !exec_started) {
        if (alu_exec_remain > 1U) {
            alu_exec_remain--;
        } else {
            alu_finish_execute();
        }
    }

    if (alu_wb_active) {
        AluWriteback wb;
        wb.reg_type = alu_inst.rd_type;
        wb.pidx = alu_inst.rd_type == TOPRandType::TREG ? alu_inst.rd_pidx[alu_wb_byte] : alu_inst.rd_pidx[0];
        wb.byte_idx = alu_wb_byte;
        uint32_t bytes = alu_inst.rd_type == TOPRandType::TMASK ? 1U : alu_dtype_bytes(alu_inst.rd_datatype);
        wb.last = (uint32_t(alu_wb_byte) + 1U) >= bytes;
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                wb.data[r][c] = uint8_t((alu_result[r][c] >> (uint32_t(alu_wb_byte) * 8U)) & 0xffU);
            }
        }
        if (aluWriteback(wb)) {
            if (wb.last) {
                alu_wb_active = false;
                alu_retire_info.reg_type = alu_inst.rd_type;
                for (int i = 0; i < 4; ++i) {
                    alu_retire_info.pidx[i] = 0;
                }
                uint32_t retire_count = alu_inst.rd_type == TOPRandType::TMASK ? 1U : alu_dtype_bytes(alu_inst.rd_datatype);
                for (uint32_t i = 0; i < retire_count; ++i) {
                    alu_retire_info.pidx[i] = alu_inst.rd_pidx[i];
                }
                alu_retire_info.valid = Int<4>((1U << retire_count) - 1U);
                alu_retire_pending = true;
            } else {
                alu_wb_byte++;
            }
        }
    }

    bool retire_sent = false;
    if (alu_retire_pending) {
        if (retireInst(alu_retire_info)) {
            alu_retire_pending = false;
            alu_busy = false;
            alu_result_ready = false;
            retire_sent = true;
        }
    }

    if (acc_busy && !acc_started) {
        if (s1FireAccget(acc_meta.pacc, acc_meta.ptemp_rd, acc_meta.dtype)) {
            acc_started = true;
            acc_send_lane = 0;
        }
    } else if (acc_busy && acc_started && acc_send_lane < (TILE_SIZE / SIMD_LANE_SIZE)) {
        Tile32SIMD chunk;
        uint32_t base_col = uint32_t(acc_send_lane) * SIMD_LANE_SIZE;
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int lane = 0; lane < SIMD_LANE_SIZE; ++lane) {
                chunk[r][lane] = alu_temps[acc_meta.ptemp_rs][r][base_col + uint32_t(lane)];
            }
        }
        sNAccOutput(acc_send_lane, chunk);
        acc_send_lane++;
    }

    if (acc_retire_pending && !retire_sent) {
        if (retireInst(acc_retire_info)) {
            acc_retire_pending = false;
            acc_busy = false;
            acc_started = false;
            acc_send_lane = 0;
        }
    }
}
