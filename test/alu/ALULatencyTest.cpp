#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/alu.hpp"

TOP("../../src/alu/ALU.hpp");
PROJECT("../../src");

REQUEST_READY(s0AluInstFire, ARG(AluInstMeta) inst);
REQUEST(sNAluMemInput, ARG(AluMemInput) in);
REQUEST_READY(s0AccgetStart, ARG(AluAccgetMeta) meta);

GLOBAL() {
uint32_t cycle = 0;
uint32_t first_wb_cycle = 0xffffffffU;
uint32_t retire_count = 0;
}

SERVICE_READY(aluWriteback, true, ARG(AluWriteback) wb) {
    if (first_wb_cycle == 0xffffffffU) {
        first_wb_cycle = cycle;
    }
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
    auto fail = [](const char *name, const char *msg) {
        std::printf("alu latency failed [%s]: %s\n", name, msg);
        std::exit(1);
    };

    auto step = [&]() {
        cycle++;
        sim_execute();
        sim_commit();
    };

    auto dtype_bytes = [](DType dtype) -> uint32_t {
        uint32_t group = (uint32_t(dtype) >> 4U) & 0x3U;
        return group == 0U ? 1U : (group == 1U ? 2U : 4U);
    };

    auto op_needs_src2 = [](AluOp op) -> bool {
        return op == AluOp::TADD || op == AluOp::TMUL;
    };

    auto f32_bits = [](float f) -> uint32_t {
        union {
            uint32_t u;
            float f;
        } v;
        v.f = f;
        return v.u;
    };

    auto fill_const_byte = [](Tile &tile, uint32_t value, uint32_t byte) {
        uint8_t data = uint8_t((value >> (byte * 8U)) & 0xffU);
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = data;
            }
        }
    };

    auto run_case = [&](const char *name, AluOp op, DType dtype, uint32_t lhs, uint32_t rhs, uint32_t expected_latency) {
        first_wb_cycle = 0xffffffffU;
        uint32_t retire_before = retire_count;

        AluInstMeta inst;
        inst.op = op;
        inst.rd_datatype = dtype;
        inst.rs1_datatype = dtype;
        inst.rs2_datatype = dtype;
        inst.rd_type = TOPRandType::TREG;
        inst.rs1_type = TOPRandType::TREG;
        inst.rs2_type = op_needs_src2(op) ? TOPRandType::TREG : TOPRandType::INVALID;
        inst.pmask = 0;
        inst.mask_enable = false;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 20 + i;
            inst.rs1_pidx[i] = 30 + i;
            inst.rs2_pidx[i] = 40 + i;
        }
        if (!s0AluInstFire(inst)) {
            fail(name, "issue rejected");
        }

        AluMemInput in;
        in.operand = 1;
        uint32_t bytes = dtype_bytes(dtype);
        for (uint32_t b = 0; b < bytes; ++b) {
            in.byte_idx = uint8_t(b);
            fill_const_byte(in.data, lhs, b);
            sNAluMemInput(in);
            step();
        }
        if (op_needs_src2(op)) {
            in.operand = 2;
            for (uint32_t b = 0; b < bytes; ++b) {
                in.byte_idx = uint8_t(b);
                fill_const_byte(in.data, rhs, b);
                sNAluMemInput(in);
                step();
            }
        }

        uint32_t start_cycle = cycle;
        for (int wait = 0; wait < 80 && first_wb_cycle == 0xffffffffU; ++wait) {
            step();
        }
        if (first_wb_cycle == 0xffffffffU) {
            fail(name, "no writeback");
        }
        if (first_wb_cycle - start_cycle != expected_latency) {
            std::printf("expected latency %u got %u\n", expected_latency, first_wb_cycle - start_cycle);
            fail(name, "latency mismatch");
        }
        for (int wait = 0; wait < 40 && retire_count == retire_before; ++wait) {
            step();
        }
        if (retire_count != retire_before + 1U) {
            fail(name, "did not retire");
        }
    };

    uint32_t chunks = (TILE_SIZE + SIMD_LANE_SIZE - 1U) / SIMD_LANE_SIZE;
    sim_reset();
    cycle = 0;
    retire_count = 0;
    step();

    run_case("uint8-add", AluOp::TADD, DType::DTYPE_UINT8, 3U, 4U, 1U + chunks - 1U);
    run_case("uint8-mul", AluOp::TMUL, DType::DTYPE_UINT8, 3U, 4U, 3U + chunks - 1U);
    run_case("f32-add", AluOp::TADD, DType::DTYPE_F32, f32_bits(1.0f), f32_bits(2.0f), 4U + chunks - 1U);
    run_case("f32-exp", AluOp::TEXP, DType::DTYPE_F32, f32_bits(1.0f), 0U, 16U + chunks - 1U);

    std::printf("alu latency passed\n");
}
