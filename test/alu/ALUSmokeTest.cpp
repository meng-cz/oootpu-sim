#include <cstdio>
#include <cstdlib>

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
bool bad = false;
}

SERVICE_READY(aluWriteback, true, ARG(AluWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
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
    auto fail = [](const char *msg) {
        std::printf("alu smoke failed: %s\n", msg);
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    auto fill_tile = [](Tile &tile, uint8_t base) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = uint8_t(base + r + c);
            }
        }
    };

    auto clear_capture = [&]() {
        for (int i = 0; i < 4; ++i) {
            captured_valid[i] = false;
            captured_pidx[i] = 0;
        }
        retire_count = 0;
        bad = false;
    };

    auto fill_const = [](Tile &tile, uint8_t value) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = value;
            }
        }
    };

    sim_reset();
    clear_capture();
    step();

    AluInstMeta inst;
    inst.op = AluOp::TADD;
    inst.rd_datatype = DType::DTYPE_UINT16;
    inst.rs1_datatype = DType::DTYPE_UINT16;
    inst.rs2_datatype = DType::DTYPE_UINT16;
    inst.rd_type = TOPRandType::TREG;
    inst.rs1_type = TOPRandType::TREG;
    inst.rs2_type = TOPRandType::TREG;
    inst.pmask = 0;
    inst.mask_enable = false;
    for (int i = 0; i < 4; ++i) {
        inst.rd_pidx[i] = 0;
        inst.rs1_pidx[i] = 0;
        inst.rs2_pidx[i] = 0;
    }
    inst.rd_pidx[0] = 10;
    inst.rd_pidx[1] = 11;
    inst.rs1_pidx[0] = 11;
    inst.rs2_pidx[0] = 12;

    if (!s0AluInstFire(inst)) {
        fail("ALU rejected basic inst");
    }

    AluMemInput in;
    in.operand = 1;
    in.byte_idx = 0;
    fill_tile(in.data, 1);
    sNAluMemInput(in);
    step();

    in.operand = 1;
    in.byte_idx = 1;
    fill_tile(in.data, 2);
    sNAluMemInput(in);
    step();

    in.operand = 2;
    in.byte_idx = 0;
    fill_tile(in.data, 3);
    sNAluMemInput(in);
    step();

    in.operand = 2;
    in.byte_idx = 1;
    fill_tile(in.data, 4);
    sNAluMemInput(in);
    step();

    for (int i = 0; i < 24; ++i) {
        step();
    }

    if (!captured_valid[0] || !captured_valid[1]) {
        fail("missing writeback bytes");
    }
    if (captured_pidx[0] != 10 || captured_pidx[1] != 11) {
        fail("writeback physical register mismatch");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            uint32_t a = uint32_t(uint8_t(1 + r + c)) | (uint32_t(uint8_t(2 + r + c)) << 8);
            uint32_t b = uint32_t(uint8_t(3 + r + c)) | (uint32_t(uint8_t(4 + r + c)) << 8);
            uint32_t sum = (a + b) & 0xffffU;
            if (captured[0][r][c] != uint8_t(sum & 0xffU) ||
                captured[1][r][c] != uint8_t((sum >> 8) & 0xffU)) {
                bad = true;
            }
        }
    }
    if (bad) {
        fail("writeback data mismatch");
    }
    if (retire_count != 1U) {
        fail("missing retire");
    }

    clear_capture();

    inst.op = AluOp::TADD;
    inst.rd_datatype = DType::DTYPE_UINT8;
    inst.rs1_datatype = DType::DTYPE_UINT8;
    inst.rs2_datatype = DType::DTYPE_UINT8;
    inst.rd_type = TOPRandType::TTEMP;
    inst.rs1_type = TOPRandType::TREG;
    inst.rs2_type = TOPRandType::INVALID;
    inst.rd_pidx[0] = 2;
    inst.rs1_pidx[0] = 21;
    inst.rs2_pidx[0] = 0;
    inst.mask_enable = false;
    if (!s0AluInstFire(inst)) {
        fail("ALU rejected temp copy-in inst");
    }
    in.operand = 1;
    in.byte_idx = 0;
    fill_tile(in.data, 9);
    sNAluMemInput(in);
    for (int i = 0; i < 5; ++i) {
        step();
    }
    if (captured_valid[0]) {
        fail("temp copy-in unexpectedly wrote back");
    }
    if (retire_count != 1U) {
        fail("temp copy-in did not retire");
    }

    clear_capture();

    inst.rd_type = TOPRandType::TREG;
    inst.rs1_type = TOPRandType::TTEMP;
    inst.rs2_type = TOPRandType::INVALID;
    inst.rd_pidx[0] = 30;
    inst.rs1_pidx[0] = 2;
    if (!s0AluInstFire(inst)) {
        fail("ALU rejected temp copy-out inst");
    }
    for (int i = 0; i < 24; ++i) {
        step();
    }
    if (!captured_valid[0] || captured_pidx[0] != 30) {
        fail("temp copy-out missing writeback");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            if (captured[0][r][c] != uint8_t(9 + r + c)) {
                bad = true;
            }
        }
    }
    if (bad) {
        fail("temp copy-out data mismatch");
    }
    if (retire_count != 1U) {
        fail("temp copy-out did not retire");
    }

    clear_capture();

    inst.op = AluOp::TADD;
    inst.rd_datatype = DType::DTYPE_F16;
    inst.rs1_datatype = DType::DTYPE_F16;
    inst.rs2_datatype = DType::DTYPE_F16;
    inst.rd_type = TOPRandType::TREG;
    inst.rs1_type = TOPRandType::TREG;
    inst.rs2_type = TOPRandType::TREG;
    inst.rd_pidx[0] = 40;
    inst.rd_pidx[1] = 41;
    inst.rs1_pidx[0] = 42;
    inst.rs1_pidx[1] = 43;
    inst.rs2_pidx[0] = 44;
    inst.rs2_pidx[1] = 45;
    if (!s0AluInstFire(inst)) {
        fail("ALU rejected fp16 add inst");
    }
    in.operand = 1;
    in.byte_idx = 0;
    fill_const(in.data, 0x00); // 1.0 half = 0x3c00
    sNAluMemInput(in);
    step();
    in.byte_idx = 1;
    fill_const(in.data, 0x3c);
    sNAluMemInput(in);
    step();
    in.operand = 2;
    in.byte_idx = 0;
    fill_const(in.data, 0x00); // 2.0 half = 0x4000
    sNAluMemInput(in);
    step();
    in.byte_idx = 1;
    fill_const(in.data, 0x40);
    sNAluMemInput(in);
    for (int i = 0; i < 24; ++i) {
        step();
    }
    if (!captured_valid[0] || !captured_valid[1]) {
        fail("fp16 add missing writeback");
    }
    if (captured_pidx[0] != 40 || captured_pidx[1] != 41) {
        fail("fp16 add pidx mismatch");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            if (captured[0][r][c] != 0x00 || captured[1][r][c] != 0x42) {
                bad = true; // 3.0 half = 0x4200
            }
        }
    }
    if (bad) {
        fail("fp16 add data mismatch");
    }
    if (retire_count != 1U) {
        fail("fp16 add did not retire");
    }

    clear_capture();

    inst.op = AluOp::TADD;
    inst.rd_datatype = DType::DTYPE_F8E4M3;
    inst.rs1_datatype = DType::DTYPE_F8E4M3;
    inst.rs2_datatype = DType::DTYPE_F8E4M3;
    inst.rd_type = TOPRandType::TREG;
    inst.rs1_type = TOPRandType::TREG;
    inst.rs2_type = TOPRandType::TREG;
    inst.rd_pidx[0] = 50;
    inst.rs1_pidx[0] = 51;
    inst.rs2_pidx[0] = 52;
    if (!s0AluInstFire(inst)) {
        fail("ALU rejected fp8 add inst");
    }
    in.operand = 1;
    in.byte_idx = 0;
    fill_const(in.data, 0x38); // E4M3 1.0
    sNAluMemInput(in);
    step();
    in.operand = 2;
    fill_const(in.data, 0x38);
    sNAluMemInput(in);
    for (int i = 0; i < 24; ++i) {
        step();
    }
    if (!captured_valid[0] || captured_pidx[0] != 50) {
        fail("fp8 add missing writeback");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            if (captured[0][r][c] != 0x40) {
                bad = true; // E4M3 2.0
            }
        }
    }
    if (bad) {
        fail("fp8 add data mismatch");
    }
    if (retire_count != 1U) {
        fail("fp8 add did not retire");
    }

    clear_capture();

    inst.op = AluOp::TRELU6;
    inst.rd_datatype = DType::DTYPE_SINT8;
    inst.rs1_datatype = DType::DTYPE_SINT8;
    inst.rs2_datatype = DType::DTYPE_SINT8;
    inst.rd_type = TOPRandType::TREG;
    inst.rs1_type = TOPRandType::TREG;
    inst.rs2_type = TOPRandType::INVALID;
    inst.rd_pidx[0] = 60;
    inst.rs1_pidx[0] = 61;
    if (!s0AluInstFire(inst)) {
        fail("ALU rejected sint8 relu6 inst");
    }
    in.operand = 1;
    in.byte_idx = 0;
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            int value = c - 2;
            in.data[r][c] = uint8_t(value);
        }
    }
    sNAluMemInput(in);
    for (int i = 0; i < 24; ++i) {
        step();
    }
    if (!captured_valid[0] || captured_pidx[0] != 60) {
        fail("sint8 relu6 missing writeback");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            int value = c - 2;
            uint8_t expect = uint8_t(value < 0 ? 0 : (value > 6 ? 6 : value));
            if (captured[0][r][c] != expect) {
                bad = true;
            }
        }
    }
    if (bad) {
        fail("sint8 relu6 data mismatch");
    }
    if (retire_count != 1U) {
        fail("sint8 relu6 did not retire");
    }

    std::printf("alu smoke passed\n");
}
