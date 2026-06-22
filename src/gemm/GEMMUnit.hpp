#pragma once

#include "header.hpp"
#include "defhelper.hpp"

// 这个模块不可综合，没有细化具体的流水线层级，只做了相关的延迟模拟

/**
 * GEMM指令完成时退休其目标PAcc
 * @param pacc 物理Acc寄存器号
 */
REQUEST(retirePAcc, ARG(uint8_t) pacc);

/**
 * PACCGET指令向TTemp寄存器输出数据
 * @param ptemp 物理Temp寄存器号
 * @param simdlane SIMD Lane 索引
 * @param data 数据
 */
REQUEST(accOutput, ARG(uint8_t) ptemp, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data);

/**
 * PACCGET指令完成时退休其目标TTemp
 * @param ptemp 物理Temp寄存器号
 */
REQUEST(retirePTemp, ARG(uint8_t) ptemp);

ALIAS_ARRAY2(AccTile, Int<96>, TILE_SIZE, TILE_SIZE);

CONFIG(GEMM_EXEC_LEVELS, 4);
CONFIG(ACCGET_CHUNKS, TILE_SIZE / SIMD_LANE_SIZE);
CONFIG(ACCGET_OUT_SLOTS, ACCGET_CHUNKS + GEMM_OUTPUT_LATENCY + 2);
CONFIG(F32_ACC_WORK_FRAC, 160);

STRUCT(GemmInst) {
    bool valid;
    uint8_t pacc;
    DType dtype;
    bool transA;
    bool transB;
    bool gotA;
    bool gotB;
};

STRUCT(GemmPipeSlot) {
    bool valid;
    uint8_t pacc;
    DType dtype;
    bool transA;
    bool transB;
    uint32_t remain;
};

STRUCT(F32Parts) {
    bool sign;
    bool zero;
    bool inf;
    bool nan;
    int exponent;
    uint32_t sig24;
};

HELPER() {
    // 这是不可综合的，先这么写，仅用于仿真
    // 其他控制信号还是用REGISTER处理吧，这里只放大块的数据用于加速
    std::array<AccTile, PACC_NUM> paccs; // 模拟PAcc寄存器文件
    Tile inputBufferA; // 模拟输入缓冲区
    Tile inputBufferB; // 模拟输入缓冲区
    std::array<Tile, GEMM_LANE_SIZE> laneBufferA;
    std::array<Tile, GEMM_LANE_SIZE> laneBufferB;
    std::array<Tile32SIMD, ACCGET_OUT_SLOTS> accgetOutputData;

inline Int<96> int_to_fixed48(int64_t v) {
    return Int<96>(v) << 48;
}

inline int64_t sign_extend_u8(uint8_t v) {
    return static_cast<int64_t>(static_cast<int8_t>(v));
}

inline Int<96> fp8_to_fixed48(uint8_t bits, int exp_bits, int mant_bits, int bias) {
    uint8_t sign = bits >> 7;
    uint8_t exp_mask = static_cast<uint8_t>((1U << exp_bits) - 1U);
    uint8_t exp = static_cast<uint8_t>((bits >> mant_bits) & exp_mask);
    uint8_t mant = static_cast<uint8_t>(bits & ((1U << mant_bits) - 1U));

    if (exp == exp_mask) {
        // E5M2 uses this for Inf/NaN; E4M3 usually reserves only NaN encodings.
        // Clamp to the largest finite magnitude that still fits the model.
        exp = static_cast<uint8_t>(exp_mask - 1U);
        mant = static_cast<uint8_t>((1U << mant_bits) - 1U);
    }

    int significand = mant;
    int exponent = 1 - bias - mant_bits;
    if (exp != 0U) {
        significand = (1 << mant_bits) | mant;
        exponent = static_cast<int>(exp) - bias - mant_bits;
    }

    Int<96> value = significand;
    int shift = exponent + 48;
    if (shift >= 0) {
        value = value << shift;
    } else {
        value = value >> -shift;
    }
    return sign != 0U ? Int<96>(-value) : value;
}

inline Int<96> input_to_fixed48(uint8_t value, DType dtype) {
    if (dtype == DType::DTYPE_SINT8) {
        return int_to_fixed48(sign_extend_u8(value));
    }
    if (dtype == DType::DTYPE_UINT8) {
        return int_to_fixed48(static_cast<int64_t>(value));
    }
    if (dtype == DType::DTYPE_F8E5M2) {
        return fp8_to_fixed48(value, 5, 2, 15);
    }
    return fp8_to_fixed48(value, 4, 3, 7);
}

inline Int<96> fixed_mul48(Int<96> lhs, Int<96> rhs) {
    Int<192> product = lhs.sint() * rhs.sint();
    return Int<96>(product.sint() >> 48);
}

inline Int<96> round_fixed48_to_int(Int<96> value) {
    bool neg = value.at<95>();
    Int<96> mag = neg ? (-value) : value;
    Int<96> whole = mag >> 48;
    Int<48> frac = mag.at<47, 0>();
    Int<48> half = Int<48>(1) << 47;
    if (frac > half || (frac == half && static_cast<bool>(whole.template at<0>()))) {
        whole = Int<96>(whole + 1U);
    }
    return neg ? (-whole) : whole;
}

inline uint32_t saturate_int_to_s32(Int<96> value) {
    if (value.sint() > Int<96>(INT32_MAX).sint()) {
        return static_cast<uint32_t>(INT32_MAX);
    }
    if (value.sint() < Int<96>(INT32_MIN).sint()) {
        return static_cast<uint32_t>(static_cast<int32_t>(INT32_MIN));
    }
    return static_cast<uint32_t>(value.template to<int32_t>());
}

inline uint32_t saturate_int_to_u32(Int<96> value) {
    if (value.sint() < Int<96>(0).sint()) {
        return 0;
    }
    if (value > Int<96>(UINT32_MAX)) {
        return UINT32_MAX;
    }
    return value.template to<uint32_t>();
}

inline Int<96> tile32_to_fixed48(uint32_t value, DType dtype) {
    if (dtype == DType::DTYPE_UINT32) {
        return Int<96>(value) << 48;
    }
    return int_to_fixed48(static_cast<int64_t>(static_cast<int32_t>(value)));
}

inline int highest_set_bit96(Int<96> value) {
    for (int i = 95; i >= 0; --i) {
        if (static_cast<bool>(value.pick(i))) {
            return i;
        }
    }
    return -1;
}

inline int highest_set_bit256(Int<256> value) {
    for (int i = 255; i >= 0; --i) {
        if (static_cast<bool>(value.pick(i))) {
            return i;
        }
    }
    return -1;
}

inline F32Parts parse_f32(uint32_t bits) {
    F32Parts p;
    p.sign = (bits & 0x80000000U) != 0U;
    p.zero = false;
    p.inf = false;
    p.nan = false;
    p.exponent = 0;
    p.sig24 = 0;

    uint32_t exp = (bits >> 23) & 0xffU;
    uint32_t mant = bits & 0x007fffffU;
    if (exp == 0xffU) {
        p.inf = mant == 0U;
        p.nan = mant != 0U;
        return p;
    }
    if (exp == 0U && mant == 0U) {
        p.zero = true;
        return p;
    }
    if (exp != 0U) {
        p.exponent = static_cast<int>(exp) - 127;
        p.sig24 = 0x00800000U | mant;
        return p;
    }

    int msb = -1;
    for (int i = 22; i >= 0; --i) {
        if ((mant & (1U << i)) != 0U) {
            msb = i;
            break;
        }
    }
    p.exponent = msb - 149;
    p.sig24 = mant << (23 - msb);
    return p;
}

inline Int<256> shift_into_f32_work(Int<256> value, int shift) {
    if (shift >= 0) {
        return value << shift;
    }

    int rshift = -shift;
    if (rshift >= 256) {
        return value == Int<256>(0) ? Int<256>(0) : Int<256>(1);
    }

    Int<256> shifted = value >> rshift;
    Int<256> truncated = Int<256>(shifted << rshift);
    Int<256> remainder = value - truncated;
    if (remainder != Int<256>(0)) {
        shifted = shifted | Int<256>(1);
    }
    return shifted;
}

inline uint32_t round_shifted_mag_to_u32(Int<256> mag, int shift) {
    if (shift >= 0) {
        return Int<32>(mag << shift).template to<uint32_t>();
    }

    int rshift = -shift;
    if (rshift >= 256) {
        return 0;
    }

    Int<256> shifted = mag >> rshift;
    uint32_t out = Int<32>(shifted.template at<31, 0>()).template to<uint32_t>();
    if (rshift == 0) {
        return out;
    }

    Int<256> truncated = Int<256>(shifted << rshift);
    Int<256> remainder = mag - truncated;
    Int<256> half = Int<256>(1) << (rshift - 1);
    if (remainder > half || (remainder == half && (out & 1U) != 0U)) {
        out++;
    }
    return out;
}

inline uint32_t f32_bits_from_work(Int<256> signed_sum, int scale_exp) {
    bool neg = static_cast<bool>(signed_sum.template at<255>());
    Int<256> mag = neg ? Int<256>(-signed_sum) : signed_sum;
    int msb = highest_set_bit256(mag);
    if (msb < 0) {
        return neg ? 0x80000000U : 0U;
    }

    int exponent = msb + scale_exp;
    int exp_field = exponent + 127;
    if (exp_field >= 255) {
        return (neg ? 0x80000000U : 0U) | 0x7f800000U;
    }

    if (exp_field <= 0) {
        uint32_t mant = round_shifted_mag_to_u32(mag, scale_exp + 149);
        if (mant >= 0x00800000U) {
            return (neg ? 0x80000000U : 0U) | 0x00800000U;
        }
        return (neg ? 0x80000000U : 0U) | (mant & 0x007fffffU);
    }

    uint32_t sig24 = round_shifted_mag_to_u32(mag, 23 - msb);
    if (sig24 >= 0x01000000U) {
        sig24 >>= 1;
        exp_field++;
        if (exp_field >= 255) {
            return (neg ? 0x80000000U : 0U) | 0x7f800000U;
        }
    }
    return (neg ? 0x80000000U : 0U) |
           (static_cast<uint32_t>(exp_field) << 23) |
           (sig24 & 0x007fffffU);
}

inline uint32_t add_fixed48_to_f32_bits(Int<96> fixed_value, uint32_t f32_bits) {
    F32Parts addend = parse_f32(f32_bits);
    if (addend.nan) {
        return f32_bits | 0x00400000U;
    }
    if (addend.inf) {
        return f32_bits;
    }

    bool fixed_neg = static_cast<bool>(fixed_value.template at<95>());
    Int<96> fixed_mag = fixed_neg ? Int<96>(-fixed_value) : fixed_value;
    int fixed_msb = highest_set_bit96(fixed_mag);
    if (fixed_msb < 0 && addend.zero) {
        return f32_bits;
    }
    if (fixed_msb < 0) {
        return f32_bits;
    }

    int fixed_exp = fixed_msb - 48;
    int common_exp = addend.zero || fixed_exp > addend.exponent ? fixed_exp : addend.exponent;
    int scale_exp = common_exp - F32_ACC_WORK_FRAC;

    Int<256> fixed_term = shift_into_f32_work(Int<256>(fixed_mag), F32_ACC_WORK_FRAC - 48 - common_exp);
    if (fixed_neg) {
        fixed_term = Int<256>(-fixed_term);
    }

    Int<256> f32_term = 0;
    if (!addend.zero) {
        f32_term = shift_into_f32_work(Int<256>(addend.sig24), F32_ACC_WORK_FRAC + addend.exponent - common_exp - 23);
        if (addend.sign) {
            f32_term = Int<256>(-f32_term);
        }
    }

    return f32_bits_from_work(Int<256>(fixed_term + f32_term), scale_exp);
}

inline uint32_t fixed48_to_tile32(Int<96> value, uint32_t addend, DType dtype) {
    if (dtype == DType::DTYPE_F32) {
        return add_fixed48_to_f32_bits(value, addend);
    }
    Int<96> sum = Int<96>(value + tile32_to_fixed48(addend, dtype));
    Int<96> rounded = round_fixed48_to_int(sum);
    if (dtype == DType::DTYPE_UINT32) {
        return saturate_int_to_u32(rounded);
    }
    return saturate_int_to_s32(rounded);
}

inline void zero_gemm_inst(GemmInst &inst) {
    inst.valid = false;
    inst.pacc = 0;
    inst.dtype = DType::DTYPE_SINT8;
    inst.transA = false;
    inst.transB = false;
    inst.gotA = false;
    inst.gotB = false;
}

inline void zero_pipe_slot(GemmPipeSlot &slot) {
    slot.valid = false;
    slot.pacc = 0;
    slot.dtype = DType::DTYPE_SINT8;
    slot.transA = false;
    slot.transB = false;
    slot.remain = 0;
}

inline GemmPipeSlot slot_from_inst(const GemmInst &inst, uint32_t remain) {
    GemmPipeSlot slot;
    slot.valid = inst.valid;
    slot.pacc = inst.pacc;
    slot.dtype = inst.dtype;
    slot.transA = inst.transA;
    slot.transB = inst.transB;
    slot.remain = remain;
    return slot;
}

inline void execute_gemm_lane(int lane, const GemmPipeSlot &slot) {
    for (int row = 0; row < TILE_SIZE; ++row) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            Int<96> sum = paccs[slot.pacc][row][col];
            for (int k = 0; k < TILE_SIZE; ++k) {
                uint8_t a = slot.transA ? laneBufferA[lane][k][row] : laneBufferA[lane][row][k];
                uint8_t b = slot.transB ? laneBufferB[lane][col][k] : laneBufferB[lane][k][col];
                sum = Int<96>(sum + fixed_mul48(input_to_fixed48(a, slot.dtype), input_to_fixed48(b, slot.dtype)));
            }
            paccs[slot.pacc][row][col] = sum;
        }
    }
}
    
}

REGISTER(feeding, GemmInst) {
    zero_gemm_inst(feeding);
}

REGISTER_ARRAY1(ready_slot, GemmPipeSlot, GEMM_LANE_SIZE, 1) {
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        zero_pipe_slot(ready_slot[i]);
    }
}

REGISTER_ARRAY1(exec1_slot, GemmPipeSlot, GEMM_LANE_SIZE, 1) {
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        zero_pipe_slot(exec1_slot[i]);
    }
}

REGISTER_ARRAY1(exec2_slot, GemmPipeSlot, GEMM_LANE_SIZE, 1) {
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        zero_pipe_slot(exec2_slot[i]);
    }
}

REGISTER_ARRAY1(exec3_slot, GemmPipeSlot, GEMM_LANE_SIZE, 1) {
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        zero_pipe_slot(exec3_slot[i]);
    }
}

REGISTER_ARRAY1(exec4_slot, GemmPipeSlot, GEMM_LANE_SIZE, 1) {
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        zero_pipe_slot(exec4_slot[i]);
    }
}

REGISTER(gemm_acc_cooldown, uint8_t) {
    gemm_acc_cooldown = 0;
}

REGISTER(accget_busy, bool) {
    accget_busy = false;
}

REGISTER(accget_accepting, bool) {
    accget_accepting = false;
}

REGISTER(accget_pacc, uint8_t) {
    accget_pacc = 0;
}

REGISTER(accget_ptemp, uint8_t) {
    accget_ptemp = 0;
}

REGISTER(accget_dtype, DType) {
    accget_dtype = DType::DTYPE_SINT32;
}

REGISTER(accget_input_count, uint8_t) {
    accget_input_count = 0;
}

REGISTER_ARRAY1(accget_out_valid, bool, ACCGET_OUT_SLOTS, 2) {
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        accget_out_valid[i] = false;
    }
}

REGISTER_ARRAY1(accget_out_ptemp, uint8_t, ACCGET_OUT_SLOTS, 2) {
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        accget_out_ptemp[i] = 0;
    }
}

REGISTER_ARRAY1(accget_out_simdlane, uint8_t, ACCGET_OUT_SLOTS, 2) {
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        accget_out_simdlane[i] = 0;
    }
}

REGISTER_ARRAY1(accget_out_remain, uint32_t, ACCGET_OUT_SLOTS, 2) {
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        accget_out_remain[i] = 0;
    }
}

/**
 * 是否可以接收一个新加入的GEMM指令
 */
QUERY(gemmFeedingReady, bool) {
    return !feeding.get().valid;
}

/**
 * 接收一个新加入的GEMM指令，使其进入等待数据状态
 * @param pacc 物理Acc寄存器号
 * @param dtype 数据类型
 * @param transA 矩阵A是否转置
 * @param transB 矩阵B是否转置
 */
SERVICE_READY(s0FireGemmInst, gemmFeedingReady(), ARG(uint8_t) pacc, ARG(DType) dtype, ARG(bool) transA, ARG(bool) transB) {
    GemmInst inst;
    inst.valid = true;
    inst.pacc = pacc;
    inst.dtype = dtype;
    inst.transA = transA;
    inst.transB = transB;
    inst.gotA = false;
    inst.gotB = false;
    feeding.setnext(inst);
}

/**
 * 在s0FireGemmInst成功后，通过两次s1s2MemInput服务将矩阵A和B的数据输入到GEMM指令中
 * @param selB 选择输入矩阵B
 * @param data 输入数据
 */
SERVICE(s1s2MemInput, ARG(bool) selB, ARG(Tile) data) {
    GemmInst inst = feeding;
    if (!inst.valid) {
        return;
    }
    if (selB) {
        inputBufferB = data;
        inst.gotB = true;
    } else {
        inputBufferA = data;
        inst.gotA = true;
    }
    feeding.setnext(inst);
}


/**
 * 是否可以接收一个新加入的PACCGET指令
 */
QUERY(accgetReady, bool) {
    return !accget_busy;
}

/**
 * 接收一个新加入的PACCGET指令，将其加入PACCGET状态机
 * @param pacc 物理Acc寄存器号
 * @param ptemp 物理Temp寄存器号
 * @param dtype 目标累加器的数据类型
 */
SERVICE_READY(s0FireAccget, accgetReady(), ARG(uint8_t) pacc, ARG(uint8_t) ptemp, ARG(DType) dtype) {
    accget_busy.setnext(true);
    accget_accepting.setnext(true);
    accget_pacc.setnext(pacc);
    accget_ptemp.setnext(ptemp);
    accget_dtype.setnext(dtype);
    accget_input_count.setnext(0);
}

/**
 * 分若干周期（TILE_SIZE / SIMD_LANE_SIZE）将TTemp中的累加器初始值输入到PACCGET状态机
 * @param simdlane 当前输入的SIMD Lane索引
 * @param data 当前输入的SIMD Lane数据
 */
SERVICE(sNAccInput, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data) {
    if (!accget_busy || !accget_accepting) {
        return;
    }

    int free_slot = -1;
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        if (!accget_out_valid[i]) {
            free_slot = i;
            break;
        }
    }
    if (free_slot < 0) {
        return;
    }

    uint32_t base_col = static_cast<uint32_t>(simdlane) * SIMD_LANE_SIZE;
    for (int row = 0; row < TILE_SIZE; ++row) {
        for (int lane = 0; lane < SIMD_LANE_SIZE; ++lane) {
            uint32_t col = base_col + static_cast<uint32_t>(lane);
            Int<96> acc = paccs[accget_pacc][row][col];
            accgetOutputData[free_slot][row][lane] = fixed48_to_tile32(acc, data[row][lane], accget_dtype);
        }
    }

    accget_out_valid.setnext<0>(free_slot, true);
    accget_out_ptemp.setnext<0>(free_slot, accget_ptemp);
    accget_out_simdlane.setnext<0>(free_slot, simdlane);
    accget_out_remain.setnext<0>(free_slot, static_cast<uint32_t>(GEMM_OUTPUT_LATENCY));

    uint8_t next_count = static_cast<uint8_t>(accget_input_count + 1U);
    accget_input_count.setnext(next_count);
    if (next_count >= ACCGET_CHUNKS) {
        accget_accepting.setnext(false);
    }
}

TICK_IMPL() {
    GemmPipeSlot next_ready[GEMM_LANE_SIZE];
    GemmPipeSlot next_exec1[GEMM_LANE_SIZE];
    GemmPipeSlot next_exec2[GEMM_LANE_SIZE];
    GemmPipeSlot next_exec3[GEMM_LANE_SIZE];
    GemmPipeSlot next_exec4[GEMM_LANE_SIZE];
    bool ready_dirty[GEMM_LANE_SIZE];
    bool exec1_dirty[GEMM_LANE_SIZE];
    bool exec2_dirty[GEMM_LANE_SIZE];
    bool exec3_dirty[GEMM_LANE_SIZE];
    bool exec4_dirty[GEMM_LANE_SIZE];
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        next_ready[i] = ready_slot[i];
        next_exec1[i] = exec1_slot[i];
        next_exec2[i] = exec2_slot[i];
        next_exec3[i] = exec3_slot[i];
        next_exec4[i] = exec4_slot[i];
        ready_dirty[i] = false;
        exec1_dirty[i] = false;
        exec2_dirty[i] = false;
        exec3_dirty[i] = false;
        exec4_dirty[i] = false;
    }

    bool retired_gemm = false;
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        if (next_exec4[i].valid && next_exec4[i].remain == 0U && !retired_gemm) {
            retirePAcc(next_exec4[i].pacc);
            zero_pipe_slot(next_exec4[i]);
            exec4_dirty[i] = true;
            retired_gemm = true;
        } else if (next_exec4[i].valid && next_exec4[i].remain > 0U) {
            next_exec4[i].remain--;
            exec4_dirty[i] = true;
        }
    }

    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        if (next_exec3[i].valid && next_exec3[i].remain == 0U && !next_exec4[i].valid) {
            next_exec4[i] = next_exec3[i];
            next_exec4[i].remain = static_cast<uint32_t>(GEMM_LAST_LEVEL_LATENCY - 1);
            zero_pipe_slot(next_exec3[i]);
            exec4_dirty[i] = true;
            exec3_dirty[i] = true;
        } else if (next_exec3[i].valid && next_exec3[i].remain > 0U) {
            next_exec3[i].remain--;
            exec3_dirty[i] = true;
        }

        if (next_exec2[i].valid && next_exec2[i].remain == 0U && !next_exec3[i].valid) {
            next_exec3[i] = next_exec2[i];
            next_exec3[i].remain = static_cast<uint32_t>(TILE_SIZE - 1);
            zero_pipe_slot(next_exec2[i]);
            exec3_dirty[i] = true;
            exec2_dirty[i] = true;
        } else if (next_exec2[i].valid && next_exec2[i].remain > 0U) {
            next_exec2[i].remain--;
            exec2_dirty[i] = true;
        }

        if (next_exec1[i].valid && next_exec1[i].remain == 0U && !next_exec2[i].valid) {
            next_exec2[i] = next_exec1[i];
            next_exec2[i].remain = static_cast<uint32_t>(TILE_SIZE - 1);
            zero_pipe_slot(next_exec1[i]);
            exec2_dirty[i] = true;
            exec1_dirty[i] = true;
        } else if (next_exec1[i].valid && next_exec1[i].remain > 0U) {
            next_exec1[i].remain--;
            exec1_dirty[i] = true;
        }
    }

    uint8_t next_cooldown = gemm_acc_cooldown;
    if (next_cooldown > 0U) {
        next_cooldown--;
    }

    if (next_cooldown == 0U) {
        for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
            if (next_ready[i].valid && !next_exec1[i].valid) {
                execute_gemm_lane(i, next_ready[i]);
                next_exec1[i] = next_ready[i];
                next_exec1[i].remain = static_cast<uint32_t>(TILE_SIZE - 1);
                zero_pipe_slot(next_ready[i]);
                exec1_dirty[i] = true;
                ready_dirty[i] = true;
                next_cooldown = 2;
                break;
            }
        }
    }

    GemmInst next_feeding = feeding;
    bool feeding_changed = false;
    if (next_feeding.valid && next_feeding.gotA && next_feeding.gotB) {
        for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
            if (!next_ready[i].valid) {
                laneBufferA[i] = inputBufferA;
                laneBufferB[i] = inputBufferB;
                next_ready[i] = slot_from_inst(next_feeding, 0);
                ready_dirty[i] = true;
                zero_gemm_inst(next_feeding);
                feeding_changed = true;
                break;
            }
        }
    }

    if (feeding_changed) {
        feeding.setnext(next_feeding);
    }
    if (next_cooldown != gemm_acc_cooldown) {
        gemm_acc_cooldown.setnext(next_cooldown);
    }
    for (int i = 0; i < GEMM_LANE_SIZE; ++i) {
        if (ready_dirty[i]) {
            ready_slot.setnext<0>(i, next_ready[i]);
        }
        if (exec1_dirty[i]) {
            exec1_slot.setnext<0>(i, next_exec1[i]);
        }
        if (exec2_dirty[i]) {
            exec2_slot.setnext<0>(i, next_exec2[i]);
        }
        if (exec3_dirty[i]) {
            exec3_slot.setnext<0>(i, next_exec3[i]);
        }
        if (exec4_dirty[i]) {
            exec4_slot.setnext<0>(i, next_exec4[i]);
        }
    }
}

TICK_IMPL() {
    bool any_accget_work = false;
    bool output_sent = false;
    for (int i = 0; i < ACCGET_OUT_SLOTS; ++i) {
        if (!accget_out_valid[i]) {
            continue;
        }
        any_accget_work = true;
        if (accget_out_remain[i] > 0U) {
            accget_out_remain.setnext<1>(i, accget_out_remain[i] - 1U);
            continue;
        }
        if (!output_sent) {
            accOutput(accget_out_ptemp[i], accget_out_simdlane[i], accgetOutputData[i]);
            accget_out_valid.setnext<1>(i, false);
            accget_out_remain.setnext<1>(i, 0);
            output_sent = true;
        }
    }

    if (accget_busy && !accget_accepting && !any_accget_work) {
        retirePTemp(accget_ptemp);
        accget_busy.setnext(false);
        accget_input_count.setnext(0);
    }
}
