#pragma once

#include "config.hpp"
#include "dtype.hpp"
#include "tile.hpp"

STRUCT(LsuLoadRowMeta) {
    uint64_t base_addr;
    uint32_t stride_elems;
    DType dtype;
    Int<PTR_WID> rd_pidx[4];
};

STRUCT(LsuStoreRowMeta) {
    uint64_t base_addr;
    uint32_t stride_elems;
    DType dtype;
};

STRUCT(LsuLoadWriteback) {
    Int<PTR_WID> pidx;
    uint8_t byte_idx;
    Tile data;
    bool last;
};

STRUCT(LsuStoreMemInput) {
    uint8_t byte_idx;
    Tile data;
};

STRUCT(LsuBusReadReq) {
    uint16_t id;
    uint64_t addr;
};

STRUCT(LsuBusReadResp) {
    uint16_t id;
    TileRow data;
};

STRUCT(LsuBusWriteReq) {
    uint64_t addr;
    TileRow data;
    TileRow byte_enable;
};

HELPER() {
inline uint64_t tpu_bank_interleave_bytes() {
    return uint64_t(TILE_SIZE);
}

inline uint32_t tpu_phys_addr_to_bank(uint64_t phys_addr) {
    return uint32_t((phys_addr / tpu_bank_interleave_bytes()) % uint64_t(LSU_BUS_BANKS));
}

inline uint64_t tpu_phys_addr_to_bank_addr(uint64_t phys_addr) {
    uint64_t chunk = phys_addr / tpu_bank_interleave_bytes();
    uint64_t offset = phys_addr % tpu_bank_interleave_bytes();
    return (chunk / uint64_t(LSU_BUS_BANKS)) * tpu_bank_interleave_bytes() + offset;
}

inline uint64_t tpu_bank_addr_to_phys_addr(uint32_t bank, uint64_t bank_addr) {
    uint64_t local_chunk = bank_addr / tpu_bank_interleave_bytes();
    uint64_t offset = bank_addr % tpu_bank_interleave_bytes();
    return (local_chunk * uint64_t(LSU_BUS_BANKS) + uint64_t(bank)) *
           tpu_bank_interleave_bytes() + offset;
}
}
