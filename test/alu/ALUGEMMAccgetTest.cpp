#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/alu.hpp"
#include "../../src/header/gemm.hpp"

TOP("../../src/alu/ALUGEMMTop.hpp");
PROJECT("../../src");

REQUEST_READY(s0AluInstFire, ARG(AluInstMeta) inst);
REQUEST(sNAluMemInput, ARG(AluMemInput) in);
REQUEST_READY(s0AccgetStart, ARG(AluAccgetMeta) meta);
REQUEST_READY(s0FireGemmInst, ARG(uint8_t) pacc, ARG(DType) dtype, ARG(bool) transA, ARG(bool) transB);
REQUEST(s1s2MemInput, ARG(bool) selB, ARG(Tile) data);
QUERY(gemmFeedingReady, bool);
QUERY(gemmAccgetReady, bool);

GLOBAL() {
Tile captured[4];
bool captured_valid[4];
uint32_t captured_pidx[4];
uint32_t writeback_count = 0;
uint32_t alu_retire_count = 0;
uint32_t gemm_retire_pacc_count = 0;
bool bad_retire = false;
}

SERVICE_READY(aluWriteback, true, ARG(AluWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
    writeback_count++;
}

SERVICE_READY(retireInst, true, ARG(AluRetireInfo) info) {
    if (info.valid != Int<4>(0)) {
        alu_retire_count++;
    }
}

SERVICE(retirePAcc, ARG(uint8_t) pacc) {
    gemm_retire_pacc_count++;
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("alu-gemm accget failed: %s\n", msg);
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

    auto fill_ones = [](Tile &tile) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = 1;
            }
        }
    };

    auto temp_seed_value = [](int r, int c) -> uint32_t {
        return uint32_t(r + c);
    };

    auto fill_temp_seed_byte = [&](Tile &tile, uint32_t byte) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint32_t value = temp_seed_value(r, c);
                tile[r][c] = uint8_t((value >> (byte * 8U)) & 0xffU);
            }
        }
    };

    auto base_inst = []() -> AluInstMeta {
        AluInstMeta inst;
        inst.op = AluOp::TADD;
        inst.rd_datatype = DType::DTYPE_SINT32;
        inst.rs1_datatype = DType::DTYPE_SINT32;
        inst.rs2_datatype = DType::DTYPE_SINT32;
        inst.rd_type = TOPRandType::TREG;
        inst.rs1_type = TOPRandType::TREG;
        inst.rs2_type = TOPRandType::INVALID;
        inst.pmask = 0;
        inst.mask_enable = false;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 0;
            inst.rs1_pidx[i] = 0;
            inst.rs2_pidx[i] = 0;
        }
        return inst;
    };

    auto issue_temp_copy_in = [&](uint8_t ptemp) {
        AluInstMeta inst = base_inst();
        inst.rd_type = TOPRandType::TTEMP;
        inst.rs1_type = TOPRandType::TREG;
        inst.rs2_type = TOPRandType::INVALID;
        inst.rd_pidx[0] = ptemp;

        uint32_t retire_before = alu_retire_count;
        for (int wait = 0; wait < 20 && !s0AluInstFire(inst); ++wait) {
            step();
        }

        AluMemInput in;
        in.operand = 1;
        for (uint32_t b = 0; b < 4; ++b) {
            in.byte_idx = uint8_t(b);
            fill_temp_seed_byte(in.data, b);
            sNAluMemInput(in);
            step();
        }

        for (int wait = 0; wait < 40 && alu_retire_count == retire_before; ++wait) {
            step();
        }
        if (alu_retire_count != retire_before + 1U) {
            fail("temp copy-in did not retire");
        }
    };

    auto issue_temp_copy_out = [&](uint8_t ptemp) {
        clear_capture();
        AluInstMeta inst = base_inst();
        inst.rd_type = TOPRandType::TREG;
        inst.rs1_type = TOPRandType::TTEMP;
        inst.rs2_type = TOPRandType::INVALID;
        inst.rs1_pidx[0] = ptemp;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 400 + i;
        }

        uint32_t retire_before = alu_retire_count;
        for (int wait = 0; wait < 20 && !s0AluInstFire(inst); ++wait) {
            step();
        }
        for (int wait = 0; wait < 80 && alu_retire_count == retire_before; ++wait) {
            step();
        }
        if (alu_retire_count != retire_before + 1U) {
            fail("temp copy-out did not retire");
        }
        if (writeback_count != 4U) {
            fail("copy-out did not write four bytes");
        }
        for (int i = 0; i < 4; ++i) {
            if (!captured_valid[i] || captured_pidx[i] != uint32_t(400 + i)) {
                fail("copy-out writeback pidx mismatch");
            }
        }
    };

    auto build_pacc = [&](uint8_t pacc) {
        Tile a;
        Tile b;
        fill_ones(a);
        fill_ones(b);

        uint32_t retire_before = gemm_retire_pacc_count;
        if (!gemmFeedingReady()) {
            fail("gemm not ready before pacc build");
        }
        if (!s0FireGemmInst(pacc, DType::DTYPE_SINT8, false, false)) {
            fail("gemm issue rejected");
        }
        step();
        s1s2MemInput(false, a);
        step();
        s1s2MemInput(true, b);
        for (int wait = 0; wait < 220 && gemm_retire_pacc_count == retire_before; ++wait) {
            step();
        }
        if (gemm_retire_pacc_count != retire_before + 1U) {
            fail("gemm pacc did not retire");
        }
    };

    auto run_accget = [&](uint8_t pacc, uint8_t ptemp_rs, uint8_t ptemp_rd) {
        AluAccgetMeta meta;
        meta.pacc = pacc;
        meta.ptemp_rs = ptemp_rs;
        meta.ptemp_rd = ptemp_rd;
        meta.dtype = DType::DTYPE_SINT32;
        meta.funct5 = 0;

        uint32_t retire_before = alu_retire_count;
        if (!gemmAccgetReady()) {
            fail("gemm accget not ready before alu accget");
        }
        for (int wait = 0; wait < 20 && !s0AccgetStart(meta); ++wait) {
            step();
        }
        for (int wait = 0; wait < 180 && alu_retire_count == retire_before; ++wait) {
            step();
        }
        if (alu_retire_count != retire_before + 1U) {
            fail("alu accget path did not retire ptemp");
        }
    };

    sim_reset();
    clear_capture();
    alu_retire_count = 0;
    gemm_retire_pacc_count = 0;
    bad_retire = false;
    step();

    uint8_t pacc = 2;
    uint8_t ptemp_rs = 1;
    uint8_t ptemp_rd = 5;

    build_pacc(pacc);
    issue_temp_copy_in(ptemp_rs);
    run_accget(pacc, ptemp_rs, ptemp_rd);
    issue_temp_copy_out(ptemp_rd);

    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            uint32_t expect = 32U + temp_seed_value(r, c);
            uint32_t got = uint32_t(captured[0][r][c]) |
                           (uint32_t(captured[1][r][c]) << 8) |
                           (uint32_t(captured[2][r][c]) << 16) |
                           (uint32_t(captured[3][r][c]) << 24);
            if (got != expect) {
                fail("copy-out data mismatch");
            }
        }
    }

    std::printf("alu-gemm accget passed\n");
}
