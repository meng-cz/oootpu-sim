#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/tmisc.hpp"

TOP("../../src/tmisc/TMISC.hpp");
PROJECT("../../src");

REQUEST_READY(s0MiscFire, ARG(TMiscInstMeta) inst);
REQUEST(miscMemInput, ARG(TMiscMemInput) in);

GLOBAL() {
Tile captured[4];
bool captured_valid[4];
uint32_t captured_pidx[4];
TOPRandType captured_type[4];
uint32_t wb_count = 0;
}

SERVICE_READY(miscWriteback, true, ARG(TMiscWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
    captured_type[wb.byte_idx] = wb.reg_type;
    wb_count++;
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("tmisc smoke failed: %s\n", msg);
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    auto clear_capture = [&]() {
        for (int i = 0; i < 4; ++i) {
            captured_valid[i] = false;
            captured_pidx[i] = 0;
            captured_type[i] = TOPRandType::INVALID;
        }
        wb_count = 0;
    };

    auto base_inst = [&]() -> TMiscInstMeta {
        TMiscInstMeta inst;
        inst.op = TMiscOp::TFILL;
        inst.rd_datatype = DType::DTYPE_UINT8;
        inst.rs1_datatype = DType::DTYPE_UINT8;
        inst.rd_type = TOPRandType::TREG;
        inst.rs1_type = TOPRandType::TREG;
        for (int i = 0; i < 4; ++i) {
            inst.rd_pidx[i] = 10 + i;
            inst.rs1_pidx[i] = 20 + i;
        }
        inst.value = 0;
        inst.row_step = 0;
        inst.col_step = 0;
        inst.step = 0;
        inst.row_off = 0;
        inst.col_off = 0;
        inst.row = 0;
        inst.col = 0;
        inst.up = 0;
        inst.bottom = 0;
        inst.left = 0;
        inst.right = 0;
        inst.mask_mode = TMiscMaskMode::RECT;
        inst.invert = false;
        return inst;
    };

    auto wait_wb = [&](uint32_t count) {
        for (int i = 0; i < 80 && wb_count < count; ++i) {
            step();
        }
        if (wb_count != count) {
            fail("writeback count mismatch");
        }
    };

    auto send_input_byte = [&](uint8_t byte_idx, uint8_t seed) {
        TMiscMemInput in;
        in.byte_idx = byte_idx;
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                in.data[r][c] = uint8_t(seed + r * 3 + c * 5);
            }
        }
        miscMemInput(in);
    };

    sim_reset();
    clear_capture();
    step();

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TFILL;
        inst.rd_datatype = DType::DTYPE_UINT16;
        inst.value = 7;
        inst.row_step = 3;
        inst.col_step = 5;
        if (!s0MiscFire(inst)) {
            fail("TFILL rejected");
        }
        wait_wb(2);
        if (!captured_valid[0] || !captured_valid[1]) {
            fail("TFILL missing bytes");
        }
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint32_t v = uint32_t(7 + r * 3 + c * 5);
                if (captured[0][r][c] != uint8_t(v & 0xffU) ||
                    captured[1][r][c] != uint8_t((v >> 8U) & 0xffU)) {
                    fail("TFILL data mismatch");
                }
            }
        }
    }

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TFILLLINE;
        inst.rd_datatype = DType::DTYPE_UINT8;
        inst.value = 2;
        inst.step = 3;
        if (!s0MiscFire(inst)) {
            fail("TFILLLINE rejected");
        }
        wait_wb(1);
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint8_t expect = uint8_t(2 + 3 * (r * TILE_SIZE + c));
                if (captured[0][r][c] != expect) {
                    fail("TFILLLINE data mismatch");
                }
            }
        }
    }

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TBCAST_R;
        inst.rd_datatype = DType::DTYPE_UINT8;
        inst.rs1_datatype = DType::DTYPE_UINT8;
        inst.col = 4;
        if (!s0MiscFire(inst)) {
            fail("TBCAST.R rejected");
        }
        send_input_byte(0, 9);
        wait_wb(1);
        for (int r = 0; r < TILE_SIZE; ++r) {
            uint8_t expect = uint8_t(9 + r * 3 + 4 * 5);
            for (int c = 0; c < TILE_SIZE; ++c) {
                if (captured[0][r][c] != expect) {
                    fail("TBCAST.R data mismatch");
                }
            }
        }
    }

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TSHIFT_Z;
        inst.rd_datatype = DType::DTYPE_UINT8;
        inst.rs1_datatype = DType::DTYPE_UINT8;
        inst.row_off = 1;
        inst.col_off = -2;
        if (!s0MiscFire(inst)) {
            fail("TSHIFT.Z rejected");
        }
        send_input_byte(0, 13);
        wait_wb(1);
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                int rr = r + 1;
                int cc = c - 2;
                uint8_t expect = 0;
                if (rr >= 0 && rr < TILE_SIZE && cc >= 0 && cc < TILE_SIZE) {
                    expect = uint8_t(13 + rr * 3 + cc * 5);
                }
                if (captured[0][r][c] != expect) {
                    fail("TSHIFT.Z data mismatch");
                }
            }
        }
    }

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TTRANS;
        inst.rd_datatype = DType::DTYPE_UINT16;
        inst.rs1_datatype = DType::DTYPE_UINT16;
        if (!s0MiscFire(inst)) {
            fail("TTRANS rejected");
        }
        send_input_byte(0, 1);
        step();
        send_input_byte(1, 2);
        wait_wb(2);
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                uint8_t lo = uint8_t(1 + c * 3 + r * 5);
                uint8_t hi = uint8_t(2 + c * 3 + r * 5);
                if (captured[0][r][c] != lo || captured[1][r][c] != hi) {
                    fail("TTRANS data mismatch");
                }
            }
        }
    }

    {
        clear_capture();
        TMiscInstMeta inst = base_inst();
        inst.op = TMiscOp::TSETMASK;
        inst.rd_type = TOPRandType::TMASK;
        inst.rd_pidx[0] = 3;
        inst.up = 2;
        inst.bottom = 3;
        inst.left = 4;
        inst.right = 5;
        inst.mask_mode = TMiscMaskMode::UT_D;
        inst.invert = false;
        if (!s0MiscFire(inst)) {
            fail("TSETMASK rejected");
        }
        wait_wb(1);
        if (captured_type[0] != TOPRandType::TMASK || captured_pidx[0] != 3U) {
            fail("TSETMASK writeback target mismatch");
        }
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                bool in_rect = r >= 2 && r < TILE_SIZE - 3 && c >= 4 && c < TILE_SIZE - 5;
                bool expect = in_rect && (c - 4 >= r - 2);
                if (captured[0][r][c] != (expect ? 1 : 0)) {
                    fail("TSETMASK data mismatch");
                }
            }
        }
    }

    std::printf("tmisc smoke passed\n");
}
