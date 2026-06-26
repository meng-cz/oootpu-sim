#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/lsu.hpp"

TOP("../../src/lsu/LSU.hpp");
PROJECT("../../src");

REQUEST_READY(s0LoadRowFire, ARG(LsuLoadRowMeta) meta, ARG(TileMask) mask, ARG(bool) masken);
REQUEST_READY(s0StoreRowFire, ARG(LsuStoreRowMeta) meta, ARG(TileMask) mask, ARG(bool) masken);
REQUEST(storeMemInput, ARG(LsuStoreMemInput) in);
REQUEST_READY(busReadResp, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadResp) resp);

GLOBAL() {
uint8_t memory[1 << 20];
bool pending_valid[LSU_BUS_BANKS][256];
uint32_t pending_delay[LSU_BUS_BANKS][256];
uint16_t pending_id[LSU_BUS_BANKS][256];
TileRow pending_data[LSU_BUS_BANKS][256];
Tile captured[4];
bool captured_valid[4];
uint32_t captured_pidx[4];
uint32_t read_req_count[LSU_BUS_BANKS];
uint32_t write_req_count[LSU_BUS_BANKS];
uint32_t load_wb_count = 0;
bool bad = false;
}

SERVICE_READY(loadRowWriteback, true, ARG(LsuLoadWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
    load_wb_count++;
}

SERVICE_READY(busReadReq, true, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadReq) req) {
    read_req_count[IDX]++;
    int free_idx = -1;
    for (int i = 0; i < 256; ++i) {
        if (!pending_valid[IDX][i] && free_idx < 0) {
            free_idx = i;
        }
    }
    if (free_idx < 0) {
        bad = true;
        return;
    }
    pending_valid[IDX][free_idx] = true;
    pending_delay[IDX][free_idx] = uint32_t(((IDX * 5U) + req.id) % 7U);
    pending_id[IDX][free_idx] = req.id;
    for (int c = 0; c < TILE_SIZE; ++c) {
        uint64_t phys_addr = tpu_bank_addr_to_phys_addr(IDX, req.addr + uint64_t(c));
        if (tpu_phys_addr_to_bank(phys_addr) != IDX) {
            bad = true;
        }
        pending_data[IDX][free_idx][c] = memory[phys_addr];
    }
}

SERVICE_READY(busWriteReq, true, ARRAY(LSU_BUS_BANKS), ARG(LsuBusWriteReq) req) {
    write_req_count[IDX]++;
    for (int c = 0; c < TILE_SIZE; ++c) {
        if (req.byte_enable[c] != 0) {
            uint64_t phys_addr = tpu_bank_addr_to_phys_addr(IDX, req.addr + uint64_t(c));
            if (tpu_phys_addr_to_bank(phys_addr) != IDX) {
                bad = true;
            }
            memory[phys_addr] = req.data[c];
        }
    }
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("lsu bank hash stress failed: %s\n", msg);
        std::exit(1);
    };

    auto dtype_bytes = [](DType dtype) -> uint32_t {
        uint32_t group = (static_cast<uint32_t>(dtype) >> 4U) & 0x3U;
        if (group == 0U) return 1U;
        if (group == 1U) return 2U;
        return 4U;
    };

    auto send_resp_bank = [&](uint32_t bank, LsuBusReadResp resp) {
        if (bank == 0) busReadResp<0>(resp);
        else if (bank == 1) busReadResp<1>(resp);
        else if (bank == 2) busReadResp<2>(resp);
        else if (bank == 3) busReadResp<3>(resp);
        else if (bank == 4) busReadResp<4>(resp);
        else if (bank == 5) busReadResp<5>(resp);
        else if (bank == 6) busReadResp<6>(resp);
        else if (bank == 7) busReadResp<7>(resp);
    };

    auto drive_responses = [&]() {
        for (uint32_t bank = 0; bank < LSU_BUS_BANKS; ++bank) {
            bool sent = false;
            for (int i = 0; i < 256; ++i) {
                if (!pending_valid[bank][i]) {
                    continue;
                }
                if (pending_delay[bank][i] > 0U) {
                    pending_delay[bank][i]--;
                    continue;
                }
                if (!sent) {
                    LsuBusReadResp resp;
                    resp.id = pending_id[bank][i];
                    for (int c = 0; c < TILE_SIZE; ++c) {
                        resp.data[c] = pending_data[bank][i][c];
                    }
                    send_resp_bank(bank, resp);
                    pending_valid[bank][i] = false;
                    sent = true;
                }
            }
        }
    };

    auto step = [&]() {
        drive_responses();
        sim_execute();
        sim_commit();
    };

    auto pattern = [](uint32_t scenario, uint32_t byte_idx, uint32_t row, uint32_t col) -> uint8_t {
        return uint8_t((scenario * 37U + byte_idx * 53U + row * 7U + col * 11U) & 0xffU);
    };

    auto mask_enabled = [](uint32_t scenario, uint32_t row, uint32_t col) -> bool {
        return ((scenario + row * 5U + col * 3U) & 3U) != 0U;
    };

    auto fill_mask = [&](TileMask &mask, uint32_t scenario) {
        for (uint32_t r = 0; r < TILE_SIZE; ++r) {
            uint32_t bits = 0;
            for (uint32_t c = 0; c < TILE_SIZE; ++c) {
                if (mask_enabled(scenario, r, c)) {
                    bits |= 1U << c;
                }
            }
            mask[r] = bits;
        }
    };

    auto fill_store_byte = [&](Tile &tile, uint32_t scenario, uint32_t byte_idx) {
        for (uint32_t r = 0; r < TILE_SIZE; ++r) {
            for (uint32_t c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = pattern(scenario, byte_idx, r, c);
            }
        }
    };

    auto clear_counts = [&]() {
        for (uint32_t b = 0; b < LSU_BUS_BANKS; ++b) {
            read_req_count[b] = 0;
            write_req_count[b] = 0;
        }
    };

    auto clear_capture = [&]() {
        for (int i = 0; i < 4; ++i) {
            captured_valid[i] = false;
            captured_pidx[i] = 0;
        }
        load_wb_count = 0;
    };

    auto count_total = [](uint32_t counts[LSU_BUS_BANKS]) -> uint32_t {
        uint32_t total = 0;
        for (uint32_t b = 0; b < LSU_BUS_BANKS; ++b) {
            total += counts[b];
        }
        return total;
    };

    auto expect_balanced_counts = [&](uint32_t counts[LSU_BUS_BANKS], uint32_t total,
                                      const char *msg) {
        if (count_total(counts) != total) {
            fail(msg);
        }
        uint32_t expect = total / LSU_BUS_BANKS;
        for (uint32_t b = 0; b < LSU_BUS_BANKS; ++b) {
            if (counts[b] != expect) {
                fail(msg);
            }
        }
    };

    auto verify_hash_roundtrip = [&]() {
        uint64_t probes[] = {
            0, 1, 31, 32, 33, 255, 256, 257, 4095, 4096,
            65535, 65536, 131071, 262143, 524287, 1048575
        };
        for (uint32_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
            uint64_t addr = probes[i];
            uint32_t bank = tpu_phys_addr_to_bank(addr);
            uint64_t bank_addr = tpu_phys_addr_to_bank_addr(addr);
            if (bank >= LSU_BUS_BANKS) {
                fail("hash bank out of range");
            }
            if (tpu_bank_addr_to_phys_addr(bank, bank_addr) != addr) {
                fail("hash probe roundtrip mismatch");
            }
        }
        for (uint64_t addr = 0; addr < (1U << 16); ++addr) {
            uint32_t bank = tpu_phys_addr_to_bank(addr);
            uint64_t bank_addr = tpu_phys_addr_to_bank_addr(addr);
            if (tpu_bank_addr_to_phys_addr(bank, bank_addr) != addr) {
                fail("hash dense roundtrip mismatch");
            }
        }
    };

    auto verify_pathological_distribution = [&](uint64_t base, uint32_t stride_elems,
                                                DType dtype, const char *msg) {
        uint32_t bytes = dtype_bytes(dtype);
        uint32_t counts[LSU_BUS_BANKS];
        for (uint32_t b = 0; b < LSU_BUS_BANKS; ++b) {
            counts[b] = 0;
        }
        for (uint32_t row = 0; row < TILE_SIZE; ++row) {
            for (uint32_t chunk = 0; chunk < bytes; ++chunk) {
                uint64_t addr = base + uint64_t(row) * uint64_t(stride_elems) *
                                uint64_t(bytes) + uint64_t(chunk) * uint64_t(TILE_SIZE);
                counts[tpu_phys_addr_to_bank(addr)]++;
            }
        }
        expect_balanced_counts(counts, TILE_SIZE * bytes, msg);
    };

    auto run_case = [&](uint32_t scenario, DType dtype, uint64_t base, uint32_t stride_elems,
                        bool expect_balanced, bool masken) {
        uint32_t bytes = dtype_bytes(dtype);
        TileMask mask;
        fill_mask(mask, scenario);
        for (uint32_t row = 0; row < TILE_SIZE; ++row) {
            uint64_t row_base = base + uint64_t(row) * uint64_t(stride_elems) *
                                uint64_t(bytes);
            for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    memory[row_base + uint64_t(col) * uint64_t(bytes) + uint64_t(byte_idx)] = 0;
                }
            }
        }
        clear_counts();
        clear_capture();

        LsuStoreRowMeta store_meta;
        store_meta.base_addr = base;
        store_meta.stride_elems = stride_elems;
        store_meta.dtype = dtype;
        if (!s0StoreRowFire(store_meta, mask, masken)) {
            fail("store rejected");
        }
        for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
            LsuStoreMemInput in;
            in.byte_idx = byte_idx;
            fill_store_byte(in.data, scenario, byte_idx);
            storeMemInput(in);
            step();
        }
        for (int i = 0; i < 512 && count_total(write_req_count) < TILE_SIZE * bytes; ++i) {
            step();
        }
        if (bad) {
            fail("bad bus write mapping");
        }
        if (expect_balanced) {
            expect_balanced_counts(write_req_count, TILE_SIZE * bytes,
                                   "balanced write bank count mismatch");
        } else if (count_total(write_req_count) != TILE_SIZE * bytes) {
            fail("write request count mismatch");
        }
        for (uint32_t row = 0; row < TILE_SIZE; ++row) {
            uint64_t row_base = base + uint64_t(row) * uint64_t(stride_elems) *
                                uint64_t(bytes);
            for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint64_t addr = row_base + uint64_t(col) * uint64_t(bytes) +
                                    uint64_t(byte_idx);
                    uint8_t expected = (!masken || mask_enabled(scenario, row, col)) ?
                                       pattern(scenario, byte_idx, row, col) : 0;
                    if (memory[addr] != expected) {
                        fail("stored memory layout mismatch");
                    }
                }
            }
        }

        clear_counts();
        clear_capture();
        LsuLoadRowMeta load_meta;
        load_meta.base_addr = base;
        load_meta.stride_elems = stride_elems;
        load_meta.dtype = dtype;
        for (int i = 0; i < 4; ++i) {
            load_meta.rd_pidx[i] = 20 + scenario * 4U + uint32_t(i);
        }
        if (!s0LoadRowFire(load_meta, mask, masken)) {
            fail("load rejected");
        }
        for (int i = 0; i < 1024 && load_wb_count < bytes; ++i) {
            step();
        }
        if (bad) {
            fail("bad bus read mapping");
        }
        if (expect_balanced) {
            expect_balanced_counts(read_req_count, TILE_SIZE * bytes,
                                   "balanced read bank count mismatch");
        } else if (count_total(read_req_count) != TILE_SIZE * bytes) {
            fail("read request count mismatch");
        }
        for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
            if (!captured_valid[byte_idx]) {
                fail("missing load writeback byte");
            }
            if (captured_pidx[byte_idx] != 20U + scenario * 4U + byte_idx) {
                fail("load writeback pidx mismatch");
            }
            for (uint32_t row = 0; row < TILE_SIZE; ++row) {
                for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                    uint8_t expected = (!masken || mask_enabled(scenario, row, col)) ?
                                       pattern(scenario, byte_idx, row, col) : 0;
                    if (captured[byte_idx][row][col] != expected) {
                        fail("load byte-plane data mismatch");
                    }
                }
            }
        }
    };

    sim_reset();
    verify_hash_roundtrip();

    for (int i = 0; i < (1 << 20); ++i) {
        memory[i] = 0;
    }
    for (int b = 0; b < LSU_BUS_BANKS; ++b) {
        for (int i = 0; i < 256; ++i) {
            pending_valid[b][i] = false;
            pending_delay[b][i] = 0;
            pending_id[b][i] = 0;
        }
    }
    clear_counts();
    clear_capture();
    bad = false;
    step();

    uint32_t pathological_stride = TILE_SIZE * LSU_BUS_BANKS;
    verify_pathological_distribution(0, pathological_stride, DType::DTYPE_UINT8,
                                     "uint8 pathological distribution mismatch");
    verify_pathological_distribution(uint64_t(TILE_SIZE * 2), pathological_stride,
                                     DType::DTYPE_UINT16,
                                     "uint16 pathological distribution mismatch");
    verify_pathological_distribution(uint64_t(TILE_SIZE * 8), pathological_stride,
                                     DType::DTYPE_UINT32,
                                     "uint32 pathological distribution mismatch");

    run_case(0, DType::DTYPE_UINT8, 0, pathological_stride, true, false);
    run_case(1, DType::DTYPE_SINT8, uint64_t(TILE_SIZE), TILE_SIZE * 3, false, false);
    run_case(2, DType::DTYPE_UINT16, uint64_t(TILE_SIZE * 2), pathological_stride, true, false);
    run_case(3, DType::DTYPE_F16, uint64_t(TILE_SIZE * 10), TILE_SIZE * 5, false, false);
    run_case(4, DType::DTYPE_UINT32, uint64_t(TILE_SIZE * 24), pathological_stride, true, false);
    run_case(5, DType::DTYPE_F32, uint64_t(TILE_SIZE * 80), TILE_SIZE * 9, false, false);
    run_case(6, DType::DTYPE_UINT16, uint64_t(TILE_SIZE * 160), pathological_stride, true, true);

    std::printf("lsu bank hash stress passed\n");
}
