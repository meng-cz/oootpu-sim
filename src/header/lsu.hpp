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

inline uint32_t tpu_bank_index_bits() {
    uint32_t bits = 0;
    uint32_t n = uint32_t(LSU_BUS_BANKS - 1);
    while (n != 0U) {
        bits++;
        n >>= 1U;
    }
    return bits;
}

inline uint64_t tpu_bank_index_mask() {
    return uint64_t(LSU_BUS_BANKS - 1);
}

inline uint32_t tpu_phys_addr_to_bank(uint64_t phys_addr) {
    uint64_t chunk = phys_addr / tpu_bank_interleave_bytes();
    uint32_t bits = tpu_bank_index_bits();
    uint64_t hashed = chunk ^ (chunk >> bits) ^ (chunk >> (2U * bits));
    return uint32_t(hashed & tpu_bank_index_mask());
}

inline uint64_t tpu_phys_addr_to_bank_addr(uint64_t phys_addr) {
    uint64_t chunk = phys_addr / tpu_bank_interleave_bytes();
    uint64_t offset = phys_addr % tpu_bank_interleave_bytes();
    return (chunk >> tpu_bank_index_bits()) * tpu_bank_interleave_bytes() + offset;
}

inline uint64_t tpu_bank_addr_to_phys_addr(uint32_t bank, uint64_t bank_addr) {
    uint64_t local_chunk = bank_addr / tpu_bank_interleave_bytes();
    uint64_t offset = bank_addr % tpu_bank_interleave_bytes();
    uint32_t bits = tpu_bank_index_bits();
    uint64_t mask = tpu_bank_index_mask();
    uint64_t mid = local_chunk & mask;
    uint64_t high = (local_chunk >> bits) & mask;
    uint64_t low = (uint64_t(bank) ^ mid ^ high) & mask;
    return ((local_chunk << bits) | low) * tpu_bank_interleave_bytes() + offset;
}
}
