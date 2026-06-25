#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/lsu.hpp"

TOP("../../src/lsu/LSU.hpp");
PROJECT("../../src");

REQUEST_READY(s0LoadRowFire, ARG(LsuLoadRowMeta) meta);
REQUEST_READY(s0StoreRowFire, ARG(LsuStoreRowMeta) meta);
REQUEST(storeMemInput, ARG(LsuStoreMemInput) in);
REQUEST_READY(busReadResp, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadResp) resp);

GLOBAL() {
uint8_t memory[8192];
bool pending_valid[LSU_BUS_BANKS][128];
uint32_t pending_delay[LSU_BUS_BANKS][128];
uint16_t pending_id[LSU_BUS_BANKS][128];
TileRow pending_data[LSU_BUS_BANKS][128];
Tile captured[4];
bool captured_valid[4];
uint32_t captured_pidx[4];
uint32_t load_wb_count = 0;
uint32_t write_req_count = 0;
bool bad = false;
}

SERVICE_READY(loadRowWriteback, true, ARG(LsuLoadWriteback) wb) {
    captured[wb.byte_idx] = wb.data;
    captured_valid[wb.byte_idx] = true;
    captured_pidx[wb.byte_idx] = wb.pidx.template to<uint32_t>();
    load_wb_count++;
}

SERVICE_READY(busReadReq, true, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadReq) req) {
    int free_idx = -1;
    for (int i = 0; i < 128; ++i) {
        if (!pending_valid[IDX][i] && free_idx < 0) {
            free_idx = i;
        }
    }
    if (free_idx < 0) {
        bad = true;
        return;
    }
    pending_valid[IDX][free_idx] = true;
    pending_delay[IDX][free_idx] = uint32_t((IDX % 3) + 1);
    pending_id[IDX][free_idx] = req.id;
    for (int c = 0; c < TILE_SIZE; ++c) {
        uint64_t phys_addr = tpu_bank_addr_to_phys_addr(IDX, req.addr + uint64_t(c));
        pending_data[IDX][free_idx][c] = memory[phys_addr];
    }
}

SERVICE_READY(busWriteReq, true, ARRAY(LSU_BUS_BANKS), ARG(LsuBusWriteReq) req) {
    write_req_count++;
    for (int c = 0; c < TILE_SIZE; ++c) {
        if (req.byte_enable[c] != 0) {
            uint64_t phys_addr = tpu_bank_addr_to_phys_addr(IDX, req.addr + uint64_t(c));
            memory[phys_addr] = req.data[c];
        }
    }
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("lsu smoke failed: %s\n", msg);
        std::exit(1);
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
            for (int i = 0; i < 128; ++i) {
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

    auto low_byte = [](int r, int c) -> uint8_t {
        return uint8_t((r * 3 + c * 5) & 0xff);
    };

    auto high_byte = [](int r, int c) -> uint8_t {
        return uint8_t(0x80 + ((r + c) & 0x3f));
    };

    auto fill_store_byte = [&](Tile &tile, uint32_t byte_idx) {
        for (int r = 0; r < TILE_SIZE; ++r) {
            for (int c = 0; c < TILE_SIZE; ++c) {
                tile[r][c] = byte_idx == 0 ? low_byte(r, c) : high_byte(r, c);
            }
        }
    };

    sim_reset();
    for (int i = 0; i < 8192; ++i) {
        memory[i] = 0;
    }
    for (int b = 0; b < LSU_BUS_BANKS; ++b) {
        for (int i = 0; i < 128; ++i) {
            pending_valid[b][i] = false;
            pending_delay[b][i] = 0;
            pending_id[b][i] = 0;
        }
    }
    for (int i = 0; i < 4; ++i) {
        captured_valid[i] = false;
        captured_pidx[i] = 0;
    }
    load_wb_count = 0;
    write_req_count = 0;
    bad = false;
    step();

    LsuStoreRowMeta store_meta;
    store_meta.base_addr = 0;
    store_meta.stride_elems = TILE_SIZE;
    store_meta.dtype = DType::DTYPE_UINT16;
    if (!s0StoreRowFire(store_meta)) {
        fail("store rejected");
    }
    LsuStoreMemInput sin;
    sin.byte_idx = 0;
    fill_store_byte(sin.data, 0);
    storeMemInput(sin);
    step();
    sin.byte_idx = 1;
    fill_store_byte(sin.data, 1);
    storeMemInput(sin);

    for (int i = 0; i < 80 && write_req_count < TILE_SIZE * 2U; ++i) {
        step();
    }
    if (bad) {
        fail("internal pending queue overflow");
    }
    if (write_req_count != TILE_SIZE * 2U) {
        fail("store write request count mismatch");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        uint64_t row_base = uint64_t(r) * uint64_t(TILE_SIZE * 2);
        for (int c = 0; c < TILE_SIZE; ++c) {
            if (memory[row_base + uint64_t(c * 2)] != low_byte(r, c) ||
                memory[row_base + uint64_t(c * 2 + 1)] != high_byte(r, c)) {
                fail("stored memory mismatch");
            }
        }
    }

    LsuLoadRowMeta load_meta;
    load_meta.base_addr = 0;
    load_meta.stride_elems = TILE_SIZE;
    load_meta.dtype = DType::DTYPE_UINT16;
    for (int i = 0; i < 4; ++i) {
        load_meta.rd_pidx[i] = 10 + i;
    }
    if (!s0LoadRowFire(load_meta)) {
        fail("load rejected");
    }
    for (int i = 0; i < 220 && load_wb_count < 2U; ++i) {
        step();
    }
    if (bad) {
        fail("bad bus behavior");
    }
    if (!captured_valid[0] || !captured_valid[1]) {
        fail("missing load writeback");
    }
    if (captured_pidx[0] != 10U || captured_pidx[1] != 11U) {
        fail("load pidx mismatch");
    }
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            if (captured[0][r][c] != low_byte(r, c) ||
                captured[1][r][c] != high_byte(r, c)) {
                fail("load data mismatch");
            }
        }
    }

    std::printf("lsu smoke passed\n");
}
