#pragma once

#include <array>
#include <cassert>
#include <cstdint>

#include "../header/lsu.hpp"

// Simulation-only Tile row-major Load/Store Unit. This model intentionally uses
// software queues and tables to keep simulation behavior readable and fast; it is
// not intended to be synthesizable.

REQUEST_READY(loadRowWriteback, ARG(LsuLoadWriteback) wb);
REQUEST_READY(busReadReq, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadReq) req);
REQUEST_READY(busWriteReq, ARRAY(LSU_BUS_BANKS), ARG(LsuBusWriteReq) req);

REGISTER(lsu_sw_ready, bool) {
    lsu_sw_ready = false;
}

HELPER() {
struct LsuReadQueueEntry {
    uint16_t id;
    uint64_t addr;
};

struct LsuWriteQueueEntry {
    uint64_t addr;
    TileRow data;
    TileRow byte_enable;
};

struct LsuLoadSlotState {
    bool valid;
    bool requests_enqueued;
    bool writeback_active;
    uint32_t recv_count;
    uint8_t wb_byte;
    LsuLoadRowMeta meta;
    std::array<Tile, 4> data;
};

struct LsuRespMapEntry {
    bool valid;
    uint32_t slot;
    uint32_t row;
    uint32_t byte_idx;
};

bool lsu_store_active;
bool lsu_store_enqueued;
LsuStoreRowMeta lsu_store_meta;
std::array<Tile, 4> lsu_store_data;
std::array<bool, 4> lsu_store_byte_valid;

std::array<LsuLoadSlotState, LSU_LOAD_SLOTS> lsu_load_slots;
std::array<LsuRespMapEntry, LSU_MAX_LOAD_OUTSTANDING> lsu_resp_map;
std::array<std::array<LsuReadQueueEntry, LSU_BANK_QUEUE_DEPTH>, LSU_BUS_BANKS> lsu_readq_data;
std::array<std::array<LsuWriteQueueEntry, LSU_BANK_QUEUE_DEPTH>, LSU_BUS_BANKS> lsu_writeq_data;
std::array<uint32_t, LSU_BUS_BANKS> lsu_readq_head;
std::array<uint32_t, LSU_BUS_BANKS> lsu_readq_tail;
std::array<uint32_t, LSU_BUS_BANKS> lsu_readq_count;
std::array<uint32_t, LSU_BUS_BANKS> lsu_writeq_head;
std::array<uint32_t, LSU_BUS_BANKS> lsu_writeq_tail;
std::array<uint32_t, LSU_BUS_BANKS> lsu_writeq_count;

inline uint32_t lsu_dtype_bytes(DType dtype) {
    uint32_t group = (static_cast<uint32_t>(dtype) >> 4U) & 0x3U;
    if (group == 0U) return 1U;
    if (group == 1U) return 2U;
    return 4U;
}

inline uint64_t lsu_row_bytes(DType dtype) {
    return uint64_t(TILE_SIZE) * uint64_t(lsu_dtype_bytes(dtype));
}

inline uint32_t lsu_total_chunks(DType dtype) {
    return TILE_SIZE * lsu_dtype_bytes(dtype);
}

inline uint32_t lsu_elements_per_bus_chunk(DType dtype) {
    return TILE_SIZE / lsu_dtype_bytes(dtype);
}

inline uint64_t lsu_chunk_addr(uint64_t base, uint32_t stride_elems, DType dtype,
                               uint32_t row, uint32_t chunk_idx) {
    uint64_t elem_bytes = lsu_dtype_bytes(dtype);
    return base + uint64_t(row) * uint64_t(stride_elems) * elem_bytes +
           uint64_t(chunk_idx) * uint64_t(TILE_SIZE);
}

inline bool lsu_aligned_meta(uint64_t base, uint32_t stride_elems, DType dtype) {
    return (base % lsu_row_bytes(dtype)) == 0U &&
           (stride_elems % uint32_t(TILE_SIZE)) == 0U;
}

inline int lsu_find_free_slot() {
    for (int i = 0; i < LSU_LOAD_SLOTS; ++i) {
        if (!lsu_load_slots[i].valid) {
            return i;
        }
    }
    return -1;
}

inline uint32_t lsu_free_resp_maps() {
    uint32_t free_count = 0;
    for (int i = 0; i < LSU_MAX_LOAD_OUTSTANDING; ++i) {
        if (!lsu_resp_map[i].valid) {
            free_count++;
        }
    }
    return free_count;
}

inline int lsu_alloc_resp_map(uint32_t slot, uint32_t row, uint32_t byte_idx) {
    for (int i = 0; i < LSU_MAX_LOAD_OUTSTANDING; ++i) {
        if (!lsu_resp_map[i].valid) {
            lsu_resp_map[i].valid = true;
            lsu_resp_map[i].slot = slot;
            lsu_resp_map[i].row = row;
            lsu_resp_map[i].byte_idx = byte_idx;
            return i;
        }
    }
    return -1;
}

inline void lsu_clear_tile(Tile &tile) {
    for (int r = 0; r < TILE_SIZE; ++r) {
        for (int c = 0; c < TILE_SIZE; ++c) {
            tile[r][c] = 0;
        }
    }
}

inline void lsu_zero_load_slot(LsuLoadSlotState &slot) {
    slot.valid = false;
    slot.requests_enqueued = false;
    slot.writeback_active = false;
    slot.recv_count = 0;
    slot.wb_byte = 0;
    slot.meta.base_addr = 0;
    slot.meta.stride_elems = TILE_SIZE;
    slot.meta.dtype = DType::DTYPE_UINT8;
    for (int i = 0; i < 4; ++i) {
        slot.meta.rd_pidx[i] = 0;
        lsu_clear_tile(slot.data[i]);
    }
}

inline void lsu_reset_sw() {
    lsu_store_active = false;
    lsu_store_enqueued = false;
    lsu_store_meta.base_addr = 0;
    lsu_store_meta.stride_elems = TILE_SIZE;
    lsu_store_meta.dtype = DType::DTYPE_UINT8;
    for (int i = 0; i < 4; ++i) {
        lsu_store_byte_valid[i] = false;
        lsu_clear_tile(lsu_store_data[i]);
    }
    for (int i = 0; i < LSU_LOAD_SLOTS; ++i) {
        lsu_zero_load_slot(lsu_load_slots[i]);
    }
    for (int i = 0; i < LSU_MAX_LOAD_OUTSTANDING; ++i) {
        lsu_resp_map[i].valid = false;
        lsu_resp_map[i].slot = 0;
        lsu_resp_map[i].row = 0;
        lsu_resp_map[i].byte_idx = 0;
    }
    for (int b = 0; b < LSU_BUS_BANKS; ++b) {
        lsu_readq_head[b] = 0;
        lsu_readq_tail[b] = 0;
        lsu_readq_count[b] = 0;
        lsu_writeq_head[b] = 0;
        lsu_writeq_tail[b] = 0;
        lsu_writeq_count[b] = 0;
    }
}

inline bool lsu_load_ready() {
    return lsu_find_free_slot() >= 0 &&
           lsu_free_resp_maps() >= lsu_total_chunks(DType::DTYPE_SINT32);
}

inline bool lsu_store_ready() {
    return !lsu_store_active;
}

inline bool lsu_store_input_done() {
    uint32_t bytes = lsu_dtype_bytes(lsu_store_meta.dtype);
    for (uint32_t b = 0; b < bytes; ++b) {
        if (!lsu_store_byte_valid[b]) {
            return false;
        }
    }
    return true;
}

inline void lsu_readq_push(uint32_t bank, const LsuReadQueueEntry &entry) {
    assert(bank < LSU_BUS_BANKS);
    assert(lsu_readq_count[bank] < LSU_BANK_QUEUE_DEPTH);
    lsu_readq_data[bank][lsu_readq_tail[bank]] = entry;
    lsu_readq_tail[bank] = (lsu_readq_tail[bank] + 1U) % LSU_BANK_QUEUE_DEPTH;
    lsu_readq_count[bank]++;
}

inline LsuReadQueueEntry lsu_readq_front(uint32_t bank) {
    assert(bank < LSU_BUS_BANKS && lsu_readq_count[bank] > 0U);
    return lsu_readq_data[bank][lsu_readq_head[bank]];
}

inline void lsu_readq_pop(uint32_t bank) {
    assert(bank < LSU_BUS_BANKS && lsu_readq_count[bank] > 0U);
    lsu_readq_head[bank] = (lsu_readq_head[bank] + 1U) % LSU_BANK_QUEUE_DEPTH;
    lsu_readq_count[bank]--;
}

inline void lsu_writeq_push(uint32_t bank, const LsuWriteQueueEntry &entry) {
    assert(bank < LSU_BUS_BANKS);
    assert(lsu_writeq_count[bank] < LSU_BANK_QUEUE_DEPTH);
    lsu_writeq_data[bank][lsu_writeq_tail[bank]] = entry;
    lsu_writeq_tail[bank] = (lsu_writeq_tail[bank] + 1U) % LSU_BANK_QUEUE_DEPTH;
    lsu_writeq_count[bank]++;
}

inline LsuWriteQueueEntry lsu_writeq_front(uint32_t bank) {
    assert(bank < LSU_BUS_BANKS && lsu_writeq_count[bank] > 0U);
    return lsu_writeq_data[bank][lsu_writeq_head[bank]];
}

inline void lsu_writeq_pop(uint32_t bank) {
    assert(bank < LSU_BUS_BANKS && lsu_writeq_count[bank] > 0U);
    lsu_writeq_head[bank] = (lsu_writeq_head[bank] + 1U) % LSU_BANK_QUEUE_DEPTH;
    lsu_writeq_count[bank]--;
}

inline void lsu_enqueue_load_requests(uint32_t slot_id) {
    LsuLoadSlotState &slot = lsu_load_slots[slot_id];
    uint32_t bytes = lsu_dtype_bytes(slot.meta.dtype);
    uint32_t chunks_per_row = bytes;
    for (uint32_t row = 0; row < TILE_SIZE; ++row) {
        for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_row; ++chunk_idx) {
            int id = lsu_alloc_resp_map(slot_id, row, chunk_idx);
            assert(id >= 0);
            uint64_t addr = lsu_chunk_addr(slot.meta.base_addr, slot.meta.stride_elems,
                                           slot.meta.dtype, row, chunk_idx);
            uint32_t bank = tpu_phys_addr_to_bank(addr);
            LsuReadQueueEntry e;
            e.id = uint16_t(id);
            e.addr = tpu_phys_addr_to_bank_addr(addr);
            lsu_readq_push(bank, e);
        }
    }
    slot.requests_enqueued = true;
}

inline void lsu_enqueue_store_requests() {
    uint32_t bytes = lsu_dtype_bytes(lsu_store_meta.dtype);
    uint32_t chunks_per_row = bytes;
    uint32_t elems_per_chunk = lsu_elements_per_bus_chunk(lsu_store_meta.dtype);
    for (uint32_t row = 0; row < TILE_SIZE; ++row) {
        for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_row; ++chunk_idx) {
            uint64_t addr = lsu_chunk_addr(lsu_store_meta.base_addr, lsu_store_meta.stride_elems,
                                           lsu_store_meta.dtype, row, chunk_idx);
            uint32_t bank = tpu_phys_addr_to_bank(addr);
            LsuWriteQueueEntry e;
            e.addr = tpu_phys_addr_to_bank_addr(addr);
            for (uint32_t local = 0; local < elems_per_chunk; ++local) {
                uint32_t col = chunk_idx * elems_per_chunk + local;
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint32_t off = local * bytes + byte_idx;
                    e.data[off] = lsu_store_data[byte_idx][row][col];
                    e.byte_enable[off] = 1;
                }
            }
            lsu_writeq_push(bank, e);
        }
    }
    lsu_store_enqueued = true;
}

inline int lsu_find_writeback_slot() {
    for (int i = 0; i < LSU_LOAD_SLOTS; ++i) {
        LsuLoadSlotState &slot = lsu_load_slots[i];
        if (!slot.valid) {
            continue;
        }
        if (slot.writeback_active) {
            return i;
        }
        if (slot.requests_enqueued && slot.recv_count >= lsu_total_chunks(slot.meta.dtype)) {
            slot.writeback_active = true;
            slot.wb_byte = 0;
            return i;
        }
    }
    return -1;
}

template<uint32_t BANK>
inline void lsu_issue_bus_bank() {
    if constexpr (BANK < LSU_BUS_BANKS) {
        if (lsu_readq_count[BANK] > 0U) {
            LsuReadQueueEntry front = lsu_readq_front(BANK);
            LsuBusReadReq req;
            req.id = front.id;
            req.addr = front.addr;
            if (busReadReq<BANK>(req)) {
                lsu_readq_pop(BANK);
            }
        }
        if (lsu_writeq_count[BANK] > 0U) {
            LsuWriteQueueEntry front = lsu_writeq_front(BANK);
            LsuBusWriteReq req;
            req.addr = front.addr;
            req.data = front.data;
            req.byte_enable = front.byte_enable;
            if (busWriteReq<BANK>(req)) {
                lsu_writeq_pop(BANK);
            }
        }
        lsu_issue_bus_bank<BANK + 1U>();
    }
}

}

SERVICE_READY(s0LoadRowFire, lsu_load_ready(), ARG(LsuLoadRowMeta) meta) {
    assert(lsu_aligned_meta(meta.base_addr, meta.stride_elems, meta.dtype));
    int slot_id = lsu_find_free_slot();
    assert(slot_id >= 0);
    LsuLoadSlotState &slot = lsu_load_slots[slot_id];
    lsu_zero_load_slot(slot);
    slot.valid = true;
    slot.meta = meta;
    lsu_enqueue_load_requests(uint32_t(slot_id));
}

SERVICE_READY(s0StoreRowFire, lsu_store_ready(), ARG(LsuStoreRowMeta) meta) {
    assert(lsu_aligned_meta(meta.base_addr, meta.stride_elems, meta.dtype));
    lsu_store_active = true;
    lsu_store_enqueued = false;
    lsu_store_meta = meta;
    for (int i = 0; i < 4; ++i) {
        lsu_store_byte_valid[i] = false;
        lsu_clear_tile(lsu_store_data[i]);
    }
}

SERVICE(storeMemInput, ARG(LsuStoreMemInput) in) {
    if (!lsu_store_active || in.byte_idx >= 4U) {
        return;
    }
    uint32_t bytes = lsu_dtype_bytes(lsu_store_meta.dtype);
    if (in.byte_idx >= bytes) {
        return;
    }
    lsu_store_data[in.byte_idx] = in.data;
    lsu_store_byte_valid[in.byte_idx] = true;
}

SERVICE_READY(busReadResp, true, ARRAY(LSU_BUS_BANKS), ARG(LsuBusReadResp) resp) {
    if (resp.id >= LSU_MAX_LOAD_OUTSTANDING || !lsu_resp_map[resp.id].valid) {
        return;
    }
    LsuRespMapEntry map = lsu_resp_map[resp.id];
    if (map.slot >= LSU_LOAD_SLOTS || map.byte_idx >= 4U || !lsu_load_slots[map.slot].valid) {
        lsu_resp_map[resp.id].valid = false;
        return;
    }
    LsuLoadSlotState &slot = lsu_load_slots[map.slot];
    uint32_t bytes = lsu_dtype_bytes(slot.meta.dtype);
    uint32_t elems_per_chunk = lsu_elements_per_bus_chunk(slot.meta.dtype);
    for (uint32_t local = 0; local < elems_per_chunk; ++local) {
        uint32_t col = map.byte_idx * elems_per_chunk + local;
        for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
            uint32_t off = local * bytes + byte_idx;
            slot.data[byte_idx][map.row][col] = resp.data[off];
        }
    }
    slot.recv_count++;
    lsu_resp_map[resp.id].valid = false;
}

TICK_IMPL() {
    if (!lsu_sw_ready) {
        lsu_reset_sw();
        lsu_sw_ready.setnext(true);
    }

    if (lsu_store_active && !lsu_store_enqueued && lsu_store_input_done()) {
        lsu_enqueue_store_requests();
        lsu_store_active = false;
    }

    int wb_slot_id = lsu_find_writeback_slot();
    if (wb_slot_id >= 0) {
        LsuLoadSlotState &slot = lsu_load_slots[wb_slot_id];
        LsuLoadWriteback wb;
        wb.byte_idx = slot.wb_byte;
        wb.pidx = slot.meta.rd_pidx[slot.wb_byte];
        wb.data = slot.data[slot.wb_byte];
        wb.last = uint32_t(slot.wb_byte) + 1U >= lsu_dtype_bytes(slot.meta.dtype);
        if (loadRowWriteback(wb)) {
            if (wb.last) {
                lsu_zero_load_slot(slot);
            } else {
                slot.wb_byte++;
            }
        }
    }

    lsu_issue_bus_bank<0>();
}
