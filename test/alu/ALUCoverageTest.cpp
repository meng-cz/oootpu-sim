#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header.hpp"

TOP("../../src/alu/ALU.hpp");
PROJECT("../../src");

REQUEST_READY(s0AluInstFire, ARG(AluInstMeta) inst);
REQUEST(sNAluMemInput, ARG(AluMemInput) in);
REQUEST_READY(s0AccgetStart, ARG(AluAccgetMeta) meta);

GLOBAL() {
Tile captured[4];
bool captured_valid[4];
uint32_t captured_pidx[4];
uint32_t retire_count = 0;
uint32_t writeback_count = 0;
}

SERVICE_READY(aluWriteback, true, ARG(AluWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
    writeback_count++;
}

SERVICE_READY(retireInst, true, ARG(AluRetireInfo) info) {
    if (info.valid != Int<4>(0)) {
        retire_count++;
    }
}

SERVICE_READY(s1FireAccget, true, ARG(uint8_t) pacc, ARG(uint8_t) ptemp, ARG(DType) dtype) {
}

SERVICE(sNAccOutput, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data) {
}

SIMULATION() {
    auto fail = [](const char *name, const char *msg, int r = -1, int c = -1) {
        std::printf("alu coverage failed [%s]: %s", name, msg);
        if (r >= 0) {
            std::printf(" at (%d,%d)", r, c);
        }
        std::printf("\n");
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    auto clear_capture = [&]() {
        for (int i = 0; i < 4; ++i) {
            captured_valid[i] = false;
            captured_pidx[i] = 0xffffffffU;
        }
        writeback_count = 0;
    };

    auto dtype_bytes = [](DType dtype) -> uint32_t {
        uint32_t group = (uint32_t(dtype) >> 4U) & 0x3U;
        return group == 0U ? 1U : (group == 1U ? 2U : 4U);
    };

    auto is_signed = [](DType dtype) -> bool {
        return dtype == DType::DTYPE_SINT8 ||
               dtype == DType::DTYPE_SINT16 ||
               dtype == DType::DTYPE_SINT32;
    };

    auto is_unsigned = [](DType dtype) -> bool {
        return dtype == DType::DTYPE_UINT8 ||
               dtype == DType::DTYPE_UINT16 ||
               dtype == DType::DTYPE_UINT32;
    };

    auto is_float = [](DType dtype) -> bool {
        return dtype == DType::DTYPE_F8E4M3 ||
               dtype == DType::DTYPE_F8E5M2 ||
               dtype == DType::DTYPE_F16 ||
               dtype == DType::DTYPE_F32;
    };

    auto mask_for = [&](DType dtype) -> uint32_t {
        uint32_t bytes = dtype_bytes(dtype);
        return bytes == 1U ? 0xffU : (bytes == 2U ? 0xffffU : 0xffffffffU);
    };

    auto mask_value = [&](uint32_t value, DType dtype) -> uint32_t {
        return value & mask_for(dtype);
    };

    auto decode_int = [&](uint32_t value, DType dtype) -> int64_t {
        if (dtype == DType::DTYPE_SINT8) {
            return int64_t(int8_t(value & 0xffU));
        }
        if (dtype == DType::DTYPE_SINT16) {
            return int64_t(int16_t(value & 0xffffU));
        }
        if (dtype == DType::DTYPE_SINT32) {
            return int64_t(int32_t(value));
        }
        return int64_t(mask_value(value, dtype));
    };

    auto f32_to_u32 = [](float f) -> uint32_t {
        union {
            uint32_t u;
            float f;
        } v;
        v.f = f;
        return v.u;
    };

    auto u32_to_f32 = [](uint32_t u) -> float {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = u;
        return v.f;
    };

    auto pow2_int = [](int exp) -> float {
        float value = 1.0f;
        if (exp >= 0) {
            for (int i = 0; i < exp; ++i) value *= 2.0f;
        } else {
            for (int i = 0; i < -exp; ++i) value *= 0.5f;
        }
        return value;
    };

    auto decode_custom_float = [&](uint32_t bits, uint32_t exp_bits, uint32_t mant_bits, int bias) -> float {
        uint32_t sign = bits >> (exp_bits + mant_bits);
        uint32_t exp_mask = (1U << exp_bits) - 1U;
        uint32_t mant_mask = (1U << mant_bits) - 1U;
        uint32_t exp = (bits >> mant_bits) & exp_mask;
        uint32_t mant = bits & mant_mask;
        float mag = 0.0f;
        if (exp == 0U) {
            mag = mant == 0U ? 0.0f : (float(mant) / float(1U << mant_bits)) * pow2_int(1 - bias);
        } else if (exp == exp_mask) {
            mag = 3.402823e38f;
        } else {
            mag = (1.0f + float(mant) / float(1U << mant_bits)) * pow2_int(int(exp) - bias);
        }
        return sign != 0U ? -mag : mag;
    };

    auto encode_custom_float = [&](float value, uint32_t exp_bits, uint32_t mant_bits, int bias) -> uint32_t {
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
    };

    auto decode_float = [&](uint32_t bits, DType dtype) -> float {
        if (dtype == DType::DTYPE_F32) return u32_to_f32(bits);
        if (dtype == DType::DTYPE_F16) return decode_custom_float(bits & 0xffffU, 5, 10, 15);
        if (dtype == DType::DTYPE_F8E5M2) return decode_custom_float(bits & 0xffU, 5, 2, 15);
        return decode_custom_float(bits & 0xffU, 4, 3, 7);
    };

    auto encode_float = [&](float value, DType dtype) -> uint32_t {
        if (dtype == DType::DTYPE_F32) return f32_to_u32(value);
        if (dtype == DType::DTYPE_F16) return encode_custom_float(value, 5, 10, 15);
        if (dtype == DType::DTYPE_F8E5M2) return encode_custom_float(value, 5, 2, 15);
        return encode_custom_float(value, 4, 3, 7);
    };

    auto absf = [](float x) -> float {
        return x < 0.0f ? -x : x;
    };

    auto sqrt_approx = [](float x) -> float {
        if (x <= 0.0f) return 0.0f;
        float y = x > 1.0f ? x : 1.0f;
        for (int i = 0; i < 8; ++i) {
            y = 0.5f * (y + x / y);
        }
        return y;
    };

    auto exp_approx = [](float x) -> float {
        if (x < -16.0f) return 0.0f;
        if (x > 16.0f) x = 16.0f;
        float term = 1.0f;
        float sum = 1.0f;
        for (int i = 1; i <= 12; ++i) {
            term = term * x / float(i);
            sum += term;
        }
        return sum < 0.0f ? 0.0f : sum;
    };

    auto tanh_approx = [&](float x) -> float {
        float e2 = exp_approx(2.0f * x);
        return (e2 - 1.0f) / (e2 + 1.0f);
    };

    auto ref_binary = [&](AluOp op, uint32_t a, uint32_t b, DType dtype) -> uint32_t {
        if (is_float(dtype)) {
            float fa = decode_float(a, dtype);
            float fb = decode_float(b, dtype);
            float out = 0.0f;
            if (op == AluOp::TADD) out = fa + fb;
            else if (op == AluOp::TSUB) out = fa - fb;
            else if (op == AluOp::TMUL) out = fa * fb;
            else if (op == AluOp::TDIV) out = 0.0f;
            else if (op == AluOp::TMAX) out = fa > fb ? fa : fb;
            else if (op == AluOp::TMIN) out = fa < fb ? fa : fb;
            return encode_float(out, dtype);
        }
        uint32_t mask = mask_for(dtype);
        uint32_t shamt = b & 31U;
        uint64_t ua = uint64_t(mask_value(a, dtype));
        uint64_t ub = uint64_t(mask_value(b, dtype));
        int64_t sa = decode_int(a, dtype);
        int64_t sb = decode_int(b, dtype);
        int64_t out = 0;
        bool signed_op = is_signed(dtype);
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
    };

    auto ref_widen_mul = [&](uint32_t a, uint32_t b, DType src_dtype, DType rd_dtype) -> uint32_t {
        if (is_float(src_dtype)) {
            float out = decode_float(a, src_dtype) * decode_float(b, src_dtype);
            return encode_float(out, rd_dtype);
        }
        if (is_signed(src_dtype)) {
            int64_t out = decode_int(a, src_dtype) * decode_int(b, src_dtype);
            return uint32_t(out) & mask_for(rd_dtype);
        }
        uint64_t out = uint64_t(mask_value(a, src_dtype)) * uint64_t(mask_value(b, src_dtype));
        return uint32_t(out) & mask_for(rd_dtype);
    };

    auto ref_unary = [&](AluOp op, uint32_t a, DType dtype) -> uint32_t {
        if (is_float(dtype)) {
            float fa = decode_float(a, dtype);
            float out = fa;
            if (op == AluOp::TABS) out = absf(fa);
            else if (op == AluOp::TNEG) out = -fa;
            else if (op == AluOp::TSQRT) out = sqrt_approx(fa);
            else if (op == AluOp::TRSQRT) out = 1.0f / sqrt_approx(fa);
            else if (op == AluOp::TEXP) out = exp_approx(fa);
            else if (op == AluOp::TRCP) out = 1.0f / fa;
            else if (op == AluOp::TRELU) out = fa > 0.0f ? fa : 0.0f;
            else if (op == AluOp::TRELU6) out = fa < 0.0f ? 0.0f : (fa > 6.0f ? 6.0f : fa);
            else if (op == AluOp::TSIGMOID) out = 1.0f / (1.0f + exp_approx(-fa));
            else if (op == AluOp::TTANH) out = tanh_approx(fa);
            else if (op == AluOp::TGELU) out = 0.5f * fa * (1.0f + tanh_approx(0.79788456f * (fa + 0.044715f * fa * fa * fa)));
            return encode_float(out, dtype);
        }
        uint32_t mask = mask_for(dtype);
        int64_t sa = decode_int(a, dtype);
        uint64_t ua = uint64_t(mask_value(a, dtype));
        int64_t out = int64_t(ua);
        if (op == AluOp::TNOT) out = int64_t(~ua);
        else if (op == AluOp::TABS) out = sa < 0 ? -sa : sa;
        else if (op == AluOp::TNEG) out = -sa;
        else if (op == AluOp::TRELU) out = sa > 0 ? sa : 0;
        else if (op == AluOp::TRELU6) out = sa < 0 ? 0 : (sa > 6 ? 6 : sa);
        else if (op == AluOp::TCVT) out = sa;
        return uint32_t(out) & mask;
    };

    auto ref_compare = [&](AluOp op, uint32_t a, uint32_t b, DType dtype) -> uint32_t {
        if (is_float(dtype)) {
            float fa = decode_float(a, dtype);
            float fb = decode_float(b, dtype);
            if (op == AluOp::TMEQ) return fa == fb ? 1U : 0U;
            if (op == AluOp::TMNEQ) return fa != fb ? 1U : 0U;
            if (op == AluOp::TMLT) return fa < fb ? 1U : 0U;
            if (op == AluOp::TMGE) return fa >= fb ? 1U : 0U;
            if (op == AluOp::TMLE) return fa <= fb ? 1U : 0U;
            if (op == AluOp::TMGT) return fa > fb ? 1U : 0U;
        }
        int64_t sa = decode_int(a, dtype);
        int64_t sb = decode_int(b, dtype);
        uint32_t ua = mask_value(a, dtype);
        uint32_t ub = mask_value(b, dtype);
        bool signed_cmp = is_signed(dtype);
        if (op == AluOp::TMEQ) return ua == ub ? 1U : 0U;
        if (op == AluOp::TMNEQ) return ua != ub ? 1U : 0U;
        if (op == AluOp::TMLT) return (signed_cmp ? sa < sb : ua < ub) ? 1U : 0U;
        if (op == AluOp::TMGE) return (signed_cmp ? sa >= sb : ua >= ub) ? 1U : 0U;
        if (op == AluOp::TMLE) return (signed_cmp ? sa <= sb : ua <= ub) ? 1U : 0U;
        if (op == AluOp::TMGT) return (signed_cmp ? sa > sb : ua > ub) ? 1U : 0U;
        return 0U;
    };

    auto make_int_value = [&](DType dtype, int r, int c, int operand) -> uint32_t {
        int64_t value = 0;
        if (operand == 2) {
            value = 1 + ((r + c) & 3);
        } else if (is_signed(dtype)) {
            value = ((r * 3 + c * 5) % 15) - 7;
        } else {
            value = 3 + ((r * 7 + c * 11) % 31);
        }
        return uint32_t(value) & mask_for(dtype);
    };

    auto make_float_value = [&](DType dtype, int r, int c, int operand) -> uint32_t {
        float value = operand == 2 ? (0.5f + float((r + c) & 1)) : (1.0f + float((r + c) % 3));
        return encode_float(value, dtype);
    };

    auto make_value = [&](DType dtype, int r, int c, int operand) -> uint32_t {
        return is_float(dtype) ? make_float_value(dtype, r, c, operand) : make_int_value(dtype, r, c, operand);
    };

    auto fill_operand_byte = [&](Tile &tile, DType dtype, int operand, uint32_t byte) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint32_t value = make_value(dtype, r, c, operand);
                tile[r][c] = uint8_t((value >> (byte * 8U)) & 0xffU);
            }
        }
    };

    auto fill_mask_pattern = [](Tile &tile) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = ((r + c) & 1) == 0 ? 1 : 0;
            }
        }
    };

    auto op_needs_src2 = [](AluOp op) -> bool {
        return op == AluOp::TADD || op == AluOp::TSUB || op == AluOp::TMUL ||
               op == AluOp::TDIV || op == AluOp::TMAX || op == AluOp::TMIN ||
               op == AluOp::TAND || op == AluOp::TOR || op == AluOp::TXOR ||
               op == AluOp::TSLL || op == AluOp::TSRL || op == AluOp::TSRA ||
               op == AluOp::TMERGE || op == AluOp::TMAND || op == AluOp::TMNAND ||
               op == AluOp::TMOR || op == AluOp::TMNOR || op == AluOp::TMXOR ||
               op == AluOp::TMNXOR || op == AluOp::TMEQ || op == AluOp::TMNEQ ||
               op == AluOp::TMLT || op == AluOp::TMGE || op == AluOp::TMLE ||
               op == AluOp::TMGT;
    };

    auto op_uses_mask_sources = [](AluOp op) -> bool {
        return op == AluOp::TMMV || op == AluOp::TMNOT ||
               op == AluOp::TMAND || op == AluOp::TMNAND ||
               op == AluOp::TMOR || op == AluOp::TMNOR ||
               op == AluOp::TMXOR || op == AluOp::TMNXOR;
    };

    auto run_case = [&](const char *name, AluOp op, DType dtype, TOPRandType rd_type, bool mask_input,
                        auto expect_fn) {
        clear_capture();
        uint32_t retire_before = retire_count;

        AluInstMeta inst;
        inst.op = op;
        inst.rd_datatype = dtype;
        inst.rs1_datatype = dtype;
        inst.rs2_datatype = dtype;
        inst.rd_type = rd_type;
        inst.rs1_type = op_uses_mask_sources(op) ? TOPRandType::TMASK : TOPRandType::TREG;
        inst.rs2_type = op_needs_src2(op) ? inst.rs1_type : TOPRandType::INVALID;
        inst.pmask = 0;
        inst.mask_enable = false;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 100 + i;
            inst.rs1_pidx[i] = 120 + i;
            inst.rs2_pidx[i] = 140 + i;
        }

        for (int wait = 0; wait < 20 && !s0AluInstFire(inst); ++wait) {
            step();
        }

        AluMemInput in;
        uint32_t src_bytes = inst.rs1_type == TOPRandType::TMASK ? 1U : dtype_bytes(dtype);
        for (uint32_t b = 0; b < src_bytes; ++b) {
            in.operand = 1;
            in.byte_idx = uint8_t(b);
            fill_operand_byte(in.data, dtype, 1, b);
            sNAluMemInput(in);
            step();
        }
        if (op_needs_src2(op)) {
            for (uint32_t b = 0; b < src_bytes; ++b) {
                in.operand = 2;
                in.byte_idx = uint8_t(b);
                fill_operand_byte(in.data, dtype, 2, b);
                sNAluMemInput(in);
                step();
            }
        }
        if (mask_input) {
            in.operand = 3;
            in.byte_idx = 0;
            fill_mask_pattern(in.data);
            sNAluMemInput(in);
            step();
        }

        for (int wait = 0; wait < 80 && retire_count == retire_before; ++wait) {
            step();
        }
        if (retire_count != retire_before + 1U) {
            fail(name, "retire count mismatch");
        }

        uint32_t out_bytes = rd_type == TOPRandType::TMASK ? 1U : dtype_bytes(dtype);
        for (uint32_t b = 0; b < out_bytes; ++b) {
            if (!captured_valid[b]) {
                fail(name, "missing output byte");
            }
            if (captured_pidx[b] != (rd_type == TOPRandType::TMASK ? 100U : 100U + b)) {
                fail(name, "writeback pidx mismatch");
            }
        }
        if (writeback_count != out_bytes) {
            fail(name, "writeback byte count mismatch");
        }

        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint32_t a = make_value(dtype, r, c, 1);
                uint32_t bval = make_value(dtype, r, c, 2);
                uint32_t expect = expect_fn(op, dtype, a, bval);
                for (uint32_t byte = 0; byte < out_bytes; ++byte) {
                    uint8_t got = captured[byte][r][c];
                    uint8_t exp = uint8_t((expect >> (byte * 8U)) & 0xffU);
                    if (got != exp) {
                        fail(name, "data mismatch", r, c);
                    }
                }
            }
        }
    };

    auto run_widen_mul_case = [&](const char *name, DType src_dtype, DType rd_dtype) {
        clear_capture();
        uint32_t retire_before = retire_count;

        AluInstMeta inst;
        inst.op = AluOp::TMUL;
        inst.rd_datatype = rd_dtype;
        inst.rs1_datatype = src_dtype;
        inst.rs2_datatype = src_dtype;
        inst.rd_type = TOPRandType::TREG;
        inst.rs1_type = TOPRandType::TREG;
        inst.rs2_type = TOPRandType::TREG;
        inst.pmask = 0;
        inst.mask_enable = false;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 300 + i;
            inst.rs1_pidx[i] = 320 + i;
            inst.rs2_pidx[i] = 340 + i;
        }

        for (int wait = 0; wait < 20 && !s0AluInstFire(inst); ++wait) {
            step();
        }

        AluMemInput in;
        uint32_t src_bytes = dtype_bytes(src_dtype);
        for (uint32_t b = 0; b < src_bytes; ++b) {
            in.operand = 1;
            in.byte_idx = uint8_t(b);
            fill_operand_byte(in.data, src_dtype, 1, b);
            sNAluMemInput(in);
            step();
        }
        for (uint32_t b = 0; b < src_bytes; ++b) {
            in.operand = 2;
            in.byte_idx = uint8_t(b);
            fill_operand_byte(in.data, src_dtype, 2, b);
            sNAluMemInput(in);
            step();
        }

        for (int wait = 0; wait < 80 && retire_count == retire_before; ++wait) {
            step();
        }
        if (retire_count != retire_before + 1U) {
            fail(name, "retire count mismatch");
        }

        uint32_t out_bytes = dtype_bytes(rd_dtype);
        for (uint32_t b = 0; b < out_bytes; ++b) {
            if (!captured_valid[b]) {
                fail(name, "missing output byte");
            }
            if (captured_pidx[b] != 300U + b) {
                fail(name, "writeback pidx mismatch");
            }
        }
        if (writeback_count != out_bytes) {
            fail(name, "writeback byte count mismatch");
        }

        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint32_t a = make_value(src_dtype, r, c, 1);
                uint32_t bval = make_value(src_dtype, r, c, 2);
                uint32_t expect = ref_widen_mul(a, bval, src_dtype, rd_dtype);
                for (uint32_t byte = 0; byte < out_bytes; ++byte) {
                    uint8_t got = captured[byte][r][c];
                    uint8_t exp = uint8_t((expect >> (byte * 8U)) & 0xffU);
                    if (got != exp) {
                        fail(name, "data mismatch", r, c);
                    }
                }
            }
        }
    };

    auto expect_binary = [&](AluOp op, DType dtype, uint32_t a, uint32_t b) -> uint32_t {
        return ref_binary(op, a, b, dtype);
    };
    auto expect_unary = [&](AluOp op, DType dtype, uint32_t a, uint32_t b) -> uint32_t {
        (void)b;
        return ref_unary(op, a, dtype);
    };
    auto expect_compare = [&](AluOp op, DType dtype, uint32_t a, uint32_t b) -> uint32_t {
        return ref_compare(op, a, b, dtype);
    };
    auto expect_mask = [&](AluOp op, DType dtype, uint32_t a, uint32_t b) -> uint32_t {
        (void)dtype;
        bool ba = (a & 0xffU) != 0U;
        bool bb = (b & 0xffU) != 0U;
        bool out = false;
        if (op == AluOp::TMMV) out = ba;
        else if (op == AluOp::TMNOT) out = !ba;
        else if (op == AluOp::TMAND) out = ba && bb;
        else if (op == AluOp::TMNAND) out = !(ba && bb);
        else if (op == AluOp::TMOR) out = ba || bb;
        else if (op == AluOp::TMNOR) out = !(ba || bb);
        else if (op == AluOp::TMXOR) out = ba != bb;
        else if (op == AluOp::TMNXOR) out = ba == bb;
        return out ? 1U : 0U;
    };

    sim_reset();
    clear_capture();
    retire_count = 0;
    step();

    DType all_dtypes[] = {
        DType::DTYPE_SINT8, DType::DTYPE_UINT8, DType::DTYPE_F8E4M3, DType::DTYPE_F8E5M2,
        DType::DTYPE_SINT16, DType::DTYPE_UINT16, DType::DTYPE_F16,
        DType::DTYPE_SINT32, DType::DTYPE_UINT32, DType::DTYPE_F32
    };
    AluOp arithmetic_ops[] = {AluOp::TADD, AluOp::TSUB, AluOp::TMUL, AluOp::TDIV, AluOp::TMAX, AluOp::TMIN};
    AluOp int_ops[] = {AluOp::TAND, AluOp::TOR, AluOp::TXOR, AluOp::TSLL, AluOp::TSRL, AluOp::TSRA};
    AluOp int_unary_ops[] = {AluOp::TNOT, AluOp::TABS, AluOp::TNEG, AluOp::TRELU, AluOp::TRELU6, AluOp::TCVT};
    AluOp float_unary_ops[] = {
        AluOp::TABS, AluOp::TNEG, AluOp::TSQRT, AluOp::TRSQRT, AluOp::TEXP, AluOp::TRCP,
        AluOp::TRELU, AluOp::TRELU6, AluOp::TSIGMOID, AluOp::TTANH, AluOp::TGELU, AluOp::TCVT
    };
    AluOp compare_ops[] = {AluOp::TMEQ, AluOp::TMNEQ, AluOp::TMLT, AluOp::TMGE, AluOp::TMLE, AluOp::TMGT};
    AluOp mask_ops[] = {AluOp::TMMV, AluOp::TMNOT, AluOp::TMAND, AluOp::TMNAND, AluOp::TMOR, AluOp::TMNOR, AluOp::TMXOR, AluOp::TMNXOR};

    char name[96];
    for (DType dtype : all_dtypes) {
        for (AluOp op : arithmetic_ops) {
            std::snprintf(name, sizeof(name), "arith dtype=%u op=%u", uint32_t(dtype), uint32_t(op));
            run_case(name, op, dtype, TOPRandType::TREG, false, expect_binary);
        }
        for (AluOp op : compare_ops) {
            std::snprintf(name, sizeof(name), "compare dtype=%u op=%u", uint32_t(dtype), uint32_t(op));
            run_case(name, op, dtype, TOPRandType::TMASK, false, expect_compare);
        }
        if (is_float(dtype)) {
            for (AluOp op : float_unary_ops) {
                std::snprintf(name, sizeof(name), "float-unary dtype=%u op=%u", uint32_t(dtype), uint32_t(op));
                run_case(name, op, dtype, TOPRandType::TREG, false, expect_unary);
            }
        } else {
            for (AluOp op : int_ops) {
                std::snprintf(name, sizeof(name), "int-binary dtype=%u op=%u", uint32_t(dtype), uint32_t(op));
                run_case(name, op, dtype, TOPRandType::TREG, false, expect_binary);
            }
            for (AluOp op : int_unary_ops) {
                std::snprintf(name, sizeof(name), "int-unary dtype=%u op=%u", uint32_t(dtype), uint32_t(op));
                run_case(name, op, dtype, TOPRandType::TREG, false, expect_unary);
            }
        }
    }

    for (AluOp op : mask_ops) {
        std::snprintf(name, sizeof(name), "mask op=%u", uint32_t(op));
        run_case(name, op, DType::DTYPE_UINT8, TOPRandType::TMASK, false, expect_mask);
    }

    DType widen_src[] = {
        DType::DTYPE_SINT8, DType::DTYPE_SINT8, DType::DTYPE_SINT16,
        DType::DTYPE_UINT8, DType::DTYPE_UINT8, DType::DTYPE_UINT16,
        DType::DTYPE_F8E4M3, DType::DTYPE_F8E5M2, DType::DTYPE_F16
    };
    DType widen_dst[] = {
        DType::DTYPE_SINT16, DType::DTYPE_SINT32, DType::DTYPE_SINT32,
        DType::DTYPE_UINT16, DType::DTYPE_UINT32, DType::DTYPE_UINT32,
        DType::DTYPE_F16, DType::DTYPE_F16, DType::DTYPE_F32
    };
    for (uint32_t i = 0; i < 9; ++i) {
        if (!((is_signed(widen_src[i]) && is_signed(widen_dst[i])) ||
              (is_unsigned(widen_src[i]) && is_unsigned(widen_dst[i])) ||
              (is_float(widen_src[i]) && is_float(widen_dst[i])))) {
            fail("widen-mul", "test pair class mismatch");
        }
        if (dtype_bytes(widen_dst[i]) <= dtype_bytes(widen_src[i])) {
            fail("widen-mul", "test pair is not wider");
        }
        std::snprintf(name, sizeof(name), "widen-mul src=%u dst=%u", uint32_t(widen_src[i]), uint32_t(widen_dst[i]));
        run_widen_mul_case(name, widen_src[i], widen_dst[i]);
    }

    clear_capture();
    uint32_t retire_before = retire_count;
    AluInstMeta merge_inst;
    merge_inst.op = AluOp::TMERGE;
    merge_inst.rd_datatype = DType::DTYPE_UINT8;
    merge_inst.rs1_datatype = DType::DTYPE_UINT8;
    merge_inst.rs2_datatype = DType::DTYPE_UINT8;
    merge_inst.rd_type = TOPRandType::TREG;
    merge_inst.rs1_type = TOPRandType::TREG;
    merge_inst.rs2_type = TOPRandType::TREG;
    merge_inst.pmask = 0;
    merge_inst.mask_enable = false;
    for (int i = 0; i < 4; ++i) {
        merge_inst.rd_pidx[i] = 200 + i;
        merge_inst.rs1_pidx[i] = 210 + i;
        merge_inst.rs2_pidx[i] = 220 + i;
    }
    if (!s0AluInstFire(merge_inst)) {
        fail("merge", "fire rejected");
    }
    AluMemInput in;
    in.operand = 1;
    in.byte_idx = 0;
    fill_operand_byte(in.data, DType::DTYPE_UINT8, 1, 0);
    sNAluMemInput(in);
    step();
    in.operand = 2;
    fill_operand_byte(in.data, DType::DTYPE_UINT8, 2, 0);
    sNAluMemInput(in);
    step();
    in.operand = 3;
    fill_mask_pattern(in.data);
    sNAluMemInput(in);
    for (int wait = 0; wait < 80 && retire_count == retire_before; ++wait) {
        step();
    }
    if (retire_count != retire_before + 1U || !captured_valid[0]) {
        fail("merge", "missing retire/writeback");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            uint32_t a = make_value(DType::DTYPE_UINT8, r, c, 1);
            uint32_t b = make_value(DType::DTYPE_UINT8, r, c, 2);
            uint8_t expect = ((r + c) & 1) == 0 ? uint8_t(a) : uint8_t(b);
            if (captured[0][r][c] != expect) {
                fail("merge", "data mismatch", r, c);
            }
        }
    }

    std::printf("alu coverage passed: %u retired cases\n", retire_count);
}
