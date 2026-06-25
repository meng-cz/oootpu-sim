#include <cstdio>
#include <cstdlib>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/gemm.hpp"

TOP("../../src/gemm/GEMMUnit.hpp");
PROJECT("../../src");

GLOBAL() {
uint32_t expected_output = 0;
uint8_t expected_ptemp = 0;
uint32_t output_count = 0;
uint32_t retire_pacc_count = 0;
uint32_t retire_ptemp_count = 0;
bool bad_output = false;
}

REQUEST_READY(s0FireGemmInst, ARG(uint8_t) pacc, ARG(DType) dtype, ARG(bool) transA, ARG(bool) transB);
REQUEST(s1s2MemInput, ARG(bool) selB, ARG(Tile) data);
REQUEST_READY(s0FireAccget, ARG(uint8_t) pacc, ARG(uint8_t) ptemp, ARG(DType) dtype);
REQUEST(sNAccInput, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data);
QUERY(gemmFeedingReady, bool);
QUERY(accgetReady, bool);

SERVICE(retirePAcc, ARG(uint8_t) pacc) {
    retire_pacc_count++;
}

SERVICE(accOutput, ARG(uint8_t) ptemp, ARG(uint8_t) simdlane, ARG(Tile32SIMD) data) {
    output_count++;
    if (ptemp != expected_ptemp || simdlane >= (TILE_SIZE / SIMD_LANE_SIZE)) {
        bad_output = true;
    }
    for (int row = 0; row < TILE_SIZE; ++row) {
        for (int lane = 0; lane < SIMD_LANE_SIZE; ++lane) {
            if (data[row][lane] != expected_output) {
                bad_output = true;
            }
        }
    }
}

SERVICE(retirePTemp, ARG(uint8_t) ptemp) {
    if (ptemp == expected_ptemp) {
        retire_ptemp_count++;
    }
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("gemm unit failed: %s\n", msg);
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    auto fill_ones = [](Tile &tile) {
        for (int row = 0; row < TILE_SIZE; ++row) {
            for (int col = 0; col < TILE_SIZE; ++col) {
                tile[row][col] = 1;
            }
        }
    };

    auto fill_simd = [](Tile32SIMD &tile, uint32_t value) {
        for (int row = 0; row < TILE_SIZE; ++row) {
            for (int lane = 0; lane < SIMD_LANE_SIZE; ++lane) {
                tile[row][lane] = value;
            }
        }
    };

    auto run_case = [&](uint8_t pacc, uint8_t ptemp, DType out_dtype, uint32_t input_value, uint32_t expected) {
        Tile a;
        Tile b;
        Tile32SIMD input_tile;
        fill_ones(a);
        fill_ones(b);
        fill_simd(input_tile, input_value);

        expected_output = expected;
        expected_ptemp = ptemp;
        output_count = 0;
        retire_pacc_count = 0;
        retire_ptemp_count = 0;
        bad_output = false;

        if (!gemmFeedingReady()) {
            fail("gemm feeding not ready");
        }
        if (!s0FireGemmInst(pacc, DType::DTYPE_SINT8, false, false)) {
            fail("gemm issue rejected");
        }
        step();

        s1s2MemInput(false, a);
        step();
        s1s2MemInput(true, b);
        step();

        for (int i = 0; i < 4; ++i) {
            step();
        }

        if (!accgetReady()) {
            fail("accget not ready");
        }
        if (!s0FireAccget(pacc, ptemp, out_dtype)) {
            fail("accget issue rejected");
        }
        step();

        for (uint8_t lane = 0; lane < (TILE_SIZE / SIMD_LANE_SIZE); ++lane) {
            sNAccInput(lane, input_tile);
            step();
        }

        for (int i = 0; i < 150; ++i) {
            step();
        }

        if (bad_output) {
            fail("output mismatch");
        }
        if (output_count != (TILE_SIZE / SIMD_LANE_SIZE)) {
            fail("wrong output count");
        }
        if (retire_ptemp_count != 1U) {
            fail("ptemp did not retire once");
        }
        if (retire_pacc_count != 1U) {
            fail("pacc did not retire once");
        }
    };

    sim_reset();

    run_case(1, 3, DType::DTYPE_SINT32, 0U, 32U);
    run_case(2, 4, DType::DTYPE_F32, 0U, 0x42000000U);
    run_case(5, 6, DType::DTYPE_F32, 0x67800000U, 0x67800000U);

    std::printf("gemm unit passed\n");
}
