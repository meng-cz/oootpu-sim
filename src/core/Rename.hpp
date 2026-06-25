#pragma once

#include <array>
#include <cassert>
#include <cstdint>

#include "../header/rename.hpp"

// This module is simulation-only and intentionally not synthesizable.
// It models rename behavior with software data structures so each cycle touches
// only the operands and retire entries that are actually present.

REQUEST_READY(s1FetchInstRaw, RESP(TInstRaw) inst);

REGISTER(rename_sw_ready, bool) {
    rename_sw_ready = false;
}

REGISTER(rename_s2_valid, bool) {
    rename_s2_valid = false;
}

REGISTER(rename_s2_inst, TInst) {
    rename_s2_inst.subopcode = 0;
    rename_s2_inst.args = 0;
    rename_s2_inst.funct5 = 0;
    rename_s2_inst.mask = 0;
    rename_s2_inst.rd_datatype = DType::DTYPE_SINT8;
    rename_s2_inst.rs1_datatype = DType::DTYPE_SINT8;
    rename_s2_inst.rs2_datatype = DType::DTYPE_SINT8;
    rename_s2_inst.rd_type = TOPRandType::INVALID;
    rename_s2_inst.rs1_type = TOPRandType::INVALID;
    rename_s2_inst.rs2_type = TOPRandType::INVALID;
    rename_s2_inst.rd_vidx = 0;
    rename_s2_inst.rs1_vidx = 0;
    rename_s2_inst.rs2_vidx = 0;
    for (int i = 0; i < 4; ++i) {
        rename_s2_inst.rd_pidx[i] = 0;
        rename_s2_inst.rs1_pidx[i] = 0;
        rename_s2_inst.rs2_pidx[i] = 0;
    }
    rename_s2_inst.pmask = 0;
}

REGISTER(rename_s1_hold_valid, bool) {
    rename_s1_hold_valid = false;
}

REGISTER(rename_s1_hold_raw, TInstRaw) {
    rename_s1_hold_raw.subopcode = 0;
    rename_s1_hold_raw.args = 0;
    rename_s1_hold_raw.funct5 = 0;
    rename_s1_hold_raw.mask = 0;
    rename_s1_hold_raw.rd_datatype = DType::DTYPE_SINT8;
    rename_s1_hold_raw.rs1_datatype = DType::DTYPE_SINT8;
    rename_s1_hold_raw.rs2_datatype = DType::DTYPE_SINT8;
    rename_s1_hold_raw.rd_type = TOPRandType::INVALID;
    rename_s1_hold_raw.rs1_type = TOPRandType::INVALID;
    rename_s1_hold_raw.rs2_type = TOPRandType::INVALID;
    rename_s1_hold_raw.rd_vidx = 0;
    rename_s1_hold_raw.rs1_vidx = 0;
    rename_s1_hold_raw.rs2_vidx = 0;
}

REGISTER_MUL(retire_read_buf, RenameRetireRead, 2) {
    for (int i = 0; i < 8; ++i) {
        retire_read_buf.pidx[i] = 0;
    }
    retire_read_buf.valid = 0;
}

REGISTER_MUL(retire_write_buf, RenameRetireWrite, 2) {
    for (int i = 0; i < 4; ++i) {
        retire_write_buf.pidx[i] = 0;
    }
    retire_write_buf.valid = 0;
}

HELPER() {
std::array<uint32_t, VTREG_SIZE> rename_vt_map;
std::array<bool, PTREG_SIZE> rename_pt_mapped;
std::array<bool, PTREG_SIZE> rename_pt_free;
std::array<uint16_t, PTREG_SIZE> rename_pt_read_count;
std::array<uint16_t, PTREG_SIZE> rename_pt_write_count;
std::array<uint8_t, PMASK_NUM> rename_mask_read_count;
std::array<uint32_t, PTREG_SIZE> rename_free_ptregs;
uint32_t rename_free_head;
uint32_t rename_free_tail;
uint32_t rename_free_count;

inline uint32_t dtype_reg_count(DType dtype) {
    uint32_t group = (uint32_t(dtype) >> 4U) & 0x3U;
    if (group == 0U) {
        return 1U;
    }
    if (group == 1U) {
        return 2U;
    }
    return 4U;
}

inline void rename_sw_reset() {
    rename_free_head = 0;
    rename_free_tail = 0;
    rename_free_count = 0;
    for (uint32_t i = 0; i < VTREG_SIZE; ++i) {
        rename_vt_map[i] = i;
    }
    for (uint32_t i = 0; i < PTREG_SIZE; ++i) {
        bool initially_mapped = i < VTREG_SIZE;
        rename_pt_mapped[i] = initially_mapped;
        rename_pt_free[i] = !initially_mapped;
        rename_pt_read_count[i] = 0;
        rename_pt_write_count[i] = 0;
        if (!initially_mapped) {
            rename_free_ptregs[rename_free_tail] = i;
            rename_free_tail = (rename_free_tail + 1U) % PTREG_SIZE;
            rename_free_count++;
        }
    }
    for (uint32_t i = 0; i < PMASK_NUM; ++i) {
        rename_mask_read_count[i] = 0;
    }
}

inline bool contains_pidx(const uint32_t values[8], uint32_t count, uint32_t pidx) {
    for (uint32_t i = 0; i < count; ++i) {
        if (values[i] == pidx) {
            return true;
        }
    }
    return false;
}

inline void maybe_free_pidx(uint32_t pidx) {
    if (!rename_pt_mapped[pidx] &&
        rename_pt_read_count[pidx] == 0U &&
        rename_pt_write_count[pidx] == 0U &&
        !rename_pt_free[pidx]) {
        rename_pt_free[pidx] = true;
        assert(rename_free_count < PTREG_SIZE);
        rename_free_ptregs[rename_free_tail] = pidx;
        rename_free_tail = (rename_free_tail + 1U) % PTREG_SIZE;
        rename_free_count++;
    }
}

inline uint32_t alloc_pidx() {
    assert(rename_free_count != 0U);
    uint32_t pidx = rename_free_ptregs[rename_free_head];
    rename_free_head = (rename_free_head + 1U) % PTREG_SIZE;
    rename_free_count--;
    assert(rename_pt_free[pidx]);
    rename_pt_free[pidx] = false;
    return pidx;
}
}

SERVICE_READY(s2GetResult, rename_s2_valid, RESP(TInst) inst) {
    inst = rename_s2_inst;
    rename_s2_valid.setnext(false);
}

SERVICE_READY(retireRead, true, ARG(RenameRetireRead) info) {
    retire_read_buf.setnext<0>(info);
}

SERVICE_READY(retireWrite, true, ARG(RenameRetireWrite) info) {
    retire_write_buf.setnext<0>(info);
}

TICK_IMPL() {
    if (!rename_sw_ready) {
        rename_sw_reset();
        rename_sw_ready.setnext(true);
    }

    uint32_t retire_read_pidx[8];
    bool retire_read_dec[8];
    uint32_t retire_write_pidx[4];
    bool retire_write_dec[4];
    for (int i = 0; i < 8; ++i) {
        retire_read_pidx[i] = 0;
        retire_read_dec[i] = false;
    }
    for (int i = 0; i < 4; ++i) {
        retire_write_pidx[i] = 0;
        retire_write_dec[i] = false;
    }

    RenameRetireRead retire_read = retire_read_buf.get();
    if (retire_read.valid != Int<8>(0)) {
        for (int i = 0; i < 8; ++i) {
            if (retire_read.valid.pick(i)) {
                uint32_t pidx = retire_read.pidx[i].template to<uint32_t>();
                retire_read_pidx[i] = pidx;
                retire_read_dec[i] = rename_pt_read_count[pidx] != 0U;
            }
        }
        RenameRetireRead empty_read;
        for (int i = 0; i < 8; ++i) {
            empty_read.pidx[i] = 0;
        }
        empty_read.valid = 0;
        retire_read_buf.setnext<1>(empty_read);
    }

    RenameRetireWrite retire_write = retire_write_buf.get();
    if (retire_write.valid != Int<4>(0)) {
        for (int i = 0; i < 4; ++i) {
            if (retire_write.valid.pick(i)) {
                uint32_t pidx = retire_write.pidx[i].template to<uint32_t>();
                retire_write_pidx[i] = pidx;
                retire_write_dec[i] = rename_pt_write_count[pidx] != 0U;
            }
        }
        RenameRetireWrite empty_write;
        for (int i = 0; i < 4; ++i) {
            empty_write.pidx[i] = 0;
        }
        empty_write.valid = 0;
        retire_write_buf.setnext<1>(empty_write);
    }

    if (!rename_s2_valid) {
        TInstRaw raw;
        bool have_raw = false;
        bool raw_from_hold = rename_s1_hold_valid;
        if (rename_s1_hold_valid) {
            raw = rename_s1_hold_raw.get();
            have_raw = true;
        } else if (s1FetchInstRaw(raw)) {
            have_raw = true;
        }

        if (have_raw) {
            TInst out;
            bool can_rename = true;
            uint32_t source_pidx[8];
            uint32_t source_count = 0;
            uint32_t rd_needed_allocs = 0;
            uint32_t rd_valid_count = 0;
            uint32_t rd_vidx[4];
            uint32_t rd_old_pidx[4];
            bool rd_need_alloc[4];

            out.subopcode = raw.subopcode;
            out.args = raw.args;
            out.funct5 = raw.funct5;
            out.mask = raw.mask;
            out.rd_datatype = raw.rd_datatype;
            out.rs1_datatype = raw.rs1_datatype;
            out.rs2_datatype = raw.rs2_datatype;
            out.rd_type = raw.rd_type;
            out.rs1_type = raw.rs1_type;
            out.rs2_type = raw.rs2_type;
            out.rd_vidx = raw.rd_vidx;
            out.rs1_vidx = raw.rs1_vidx;
            out.rs2_vidx = raw.rs2_vidx;
            out.pmask = 0;
            for (int i = 0; i < 4; ++i) {
                out.rd_pidx[i] = 0;
                out.rs1_pidx[i] = 0;
                out.rs2_pidx[i] = 0;
                rd_vidx[i] = 0;
                rd_old_pidx[i] = 0;
                rd_need_alloc[i] = false;
            }

            if (raw.rs1_type == TOPRandType::TREG) {
                uint32_t cnt = dtype_reg_count(raw.rs1_datatype);
                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t vidx = raw.rs1_vidx.template to<uint32_t>() + i;
                    uint32_t pidx = rename_vt_map[vidx];
                    out.rs1_pidx[i] = pidx;
                    source_pidx[source_count++] = pidx;
                }
            } else if (raw.rs1_type == TOPRandType::TTEMP) {
                out.rs1_pidx[0] = 0;
            } else if (raw.rs1_type == TOPRandType::TACC) {
                out.rs1_pidx[0] = 0;
            } else if (raw.rs1_type == TOPRandType::TMASK) {
                uint32_t midx = raw.rs1_vidx.template to<uint32_t>() & 0x3U;
                if (midx != 0U) {
                    out.rs1_pidx[0] = midx;
                }
            }

            if (raw.rs2_type == TOPRandType::TREG) {
                uint32_t cnt = dtype_reg_count(raw.rs2_datatype);
                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t vidx = raw.rs2_vidx.template to<uint32_t>() + i;
                    uint32_t pidx = rename_vt_map[vidx];
                    out.rs2_pidx[i] = pidx;
                    source_pidx[source_count++] = pidx;
                }
            } else if (raw.rs2_type == TOPRandType::TMASK) {
                uint32_t midx = raw.rs2_vidx.template to<uint32_t>() & 0x3U;
                if (midx != 0U) {
                    out.rs2_pidx[0] = midx;
                }
            }

            if (raw.mask != Int<2>(0)) {
                out.pmask = raw.mask;
            }

            if (raw.rd_type == TOPRandType::TREG) {
                uint32_t cnt = dtype_reg_count(raw.rd_datatype);
                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t vidx = raw.rd_vidx.template to<uint32_t>() + i;
                    uint32_t old_pidx = rename_vt_map[vidx];
                    rd_vidx[rd_valid_count] = vidx;
                    rd_old_pidx[rd_valid_count] = old_pidx;
                    rd_need_alloc[rd_valid_count] =
                        rename_pt_read_count[old_pidx] != 0U ||
                        contains_pidx(source_pidx, source_count, old_pidx);
                    if (rd_need_alloc[rd_valid_count]) {
                        rd_needed_allocs++;
                    }
                    rd_valid_count++;
                }
            }

            if (rd_needed_allocs > rename_free_count) {
                can_rename = false;
            }

            if (can_rename) {
                for (uint32_t i = 0; i < source_count; ++i) {
                    rename_pt_read_count[source_pidx[i]]++;
                }

                if (raw.rs1_type == TOPRandType::TMASK) {
                    uint32_t midx = raw.rs1_vidx.template to<uint32_t>() & 0x3U;
                    if (midx != 0U) {
                        rename_mask_read_count[midx]++;
                    }
                }
                if (raw.rs2_type == TOPRandType::TMASK) {
                    uint32_t midx = raw.rs2_vidx.template to<uint32_t>() & 0x3U;
                    if (midx != 0U) {
                        rename_mask_read_count[midx]++;
                    }
                }
                if (raw.mask != Int<2>(0)) {
                    rename_mask_read_count[raw.mask.template to<uint32_t>()]++;
                }

                if (raw.rd_type == TOPRandType::TREG) {
                    for (uint32_t i = 0; i < rd_valid_count; ++i) {
                        uint32_t pidx = rd_old_pidx[i];
                        if (rd_need_alloc[i]) {
                            pidx = alloc_pidx();
                            rename_vt_map[rd_vidx[i]] = pidx;
                            rename_pt_mapped[rd_old_pidx[i]] = false;
                            maybe_free_pidx(rd_old_pidx[i]);
                            rename_pt_mapped[pidx] = true;
                        }
                        out.rd_pidx[i] = pidx;
                        rename_pt_write_count[pidx]++;
                    }
                } else if (raw.rd_type == TOPRandType::TTEMP) {
                    out.rd_pidx[0] = 0;
                } else if (raw.rd_type == TOPRandType::TACC) {
                    out.rd_pidx[0] = 0;
                } else if (raw.rd_type == TOPRandType::TMASK) {
                    uint32_t midx = raw.rd_vidx.template to<uint32_t>() & 0x3U;
                    if (midx != 0U) {
                        out.rd_pidx[0] = midx;
                    }
                }

                rename_s2_inst.setnext(out);
                rename_s2_valid.setnext(true);
                if (raw_from_hold) {
                    rename_s1_hold_valid.setnext(false);
                }
            } else if (!raw_from_hold) {
                rename_s1_hold_raw.setnext(raw);
                rename_s1_hold_valid.setnext(true);
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        if (retire_read_dec[i]) {
            uint32_t pidx = retire_read_pidx[i];
            rename_pt_read_count[pidx]--;
            maybe_free_pidx(pidx);
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (retire_write_dec[i]) {
            uint32_t pidx = retire_write_pidx[i];
            rename_pt_write_count[pidx]--;
            maybe_free_pidx(pidx);
        }
    }
}
