#include <array>
#include <cstdio>
#include <cstdlib>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header.hpp"

TOP("../../src/core/Rename.hpp");
PROJECT("../../src");

REQUEST_READY(s2GetResult, RESP(TInst) inst);
REQUEST_READY(retireRead, ARG(RenameRetireRead) info);
REQUEST_READY(retireWrite, ARG(RenameRetireWrite) info);

GLOBAL() {
bool feed_valid = false;
TInstRaw feed_inst;
}

SERVICE_READY(s1FetchInstRaw, feed_valid, RESP(TInstRaw) inst) {
    inst = feed_inst;
    feed_valid = false;
}

SIMULATION() {
    auto fail = [](const char *msg) {
        std::printf("rename stress failed: %s\n", msg);
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    auto rng_next = [](uint32_t &state) {
        state = state * 1664525U + 1013904223U;
        return state;
    };

    auto dtype_for_group = [](uint32_t group) {
        if (group == 0U) {
            return DType::DTYPE_SINT8;
        }
        if (group == 1U) {
            return DType::DTYPE_SINT16;
        }
        return DType::DTYPE_SINT32;
    };

    auto reg_count = [](DType dtype) {
        uint32_t group = (uint32_t(dtype) >> 4U) & 0x3U;
        if (group == 0U) {
            return 1U;
        }
        if (group == 1U) {
            return 2U;
        }
        return 4U;
    };

    auto clear_raw = []() {
        TInstRaw inst;
        inst.subopcode = 0x70;
        inst.args = 0;
        inst.funct5 = 0;
        inst.mask = 0;
        inst.rd_datatype = DType::DTYPE_SINT8;
        inst.rs1_datatype = DType::DTYPE_SINT8;
        inst.rs2_datatype = DType::DTYPE_SINT8;
        inst.rd_type = TOPRandType::INVALID;
        inst.rs1_type = TOPRandType::INVALID;
        inst.rs2_type = TOPRandType::INVALID;
        inst.rd_vidx = 0;
        inst.rs1_vidx = 0;
        inst.rs2_vidx = 0;
        return inst;
    };

    struct PendingRetire {
        bool valid;
        bool is_write;
        uint32_t ready_inst;
        uint32_t pidx;
    };

    struct Model {
        std::array<uint32_t, VTREG_SIZE> map;
        std::array<bool, PTREG_SIZE> mapped;
        std::array<bool, PTREG_SIZE> free;
        std::array<uint16_t, PTREG_SIZE> read_count;
        std::array<uint16_t, PTREG_SIZE> write_count;
        std::array<uint32_t, PTREG_SIZE> free_q;
        uint32_t free_head;
        uint32_t free_tail;
        uint32_t free_count;

        void reset() {
            free_head = 0;
            free_tail = 0;
            free_count = 0;
            for (uint32_t i = 0; i < VTREG_SIZE; ++i) {
                map[i] = i;
            }
            for (uint32_t i = 0; i < PTREG_SIZE; ++i) {
                bool initially_mapped = i < VTREG_SIZE;
                mapped[i] = initially_mapped;
                free[i] = !initially_mapped;
                read_count[i] = 0;
                write_count[i] = 0;
                if (!initially_mapped) {
                    free_q[free_tail] = i;
                    free_tail = (free_tail + 1U) % PTREG_SIZE;
                    free_count++;
                }
            }
        }

        bool source_used(const uint32_t srcs[8], uint32_t count, uint32_t pidx) {
            for (uint32_t i = 0; i < count; ++i) {
                if (srcs[i] == pidx) {
                    return true;
                }
            }
            return false;
        }

        void maybe_free(uint32_t pidx) {
            if (!mapped[pidx] && read_count[pidx] == 0U && write_count[pidx] == 0U && !free[pidx]) {
                free[pidx] = true;
                if (free_count >= PTREG_SIZE) {
                    std::printf("rename stress failed: model free queue overflow\n");
                    std::exit(1);
                }
                free_q[free_tail] = pidx;
                free_tail = (free_tail + 1U) % PTREG_SIZE;
                free_count++;
            }
        }

        uint32_t alloc() {
            if (free_count == 0U) {
                std::printf("rename stress failed: model free queue exhausted\n");
                std::exit(1);
            }
            uint32_t pidx = free_q[free_head];
            free_head = (free_head + 1U) % PTREG_SIZE;
            free_count--;
            if (!free[pidx]) {
                std::printf("rename stress failed: model allocated non-free pidx\n");
                std::exit(1);
            }
            free[pidx] = false;
            return pidx;
        }

        void retire_read(uint32_t pidx) {
            if (read_count[pidx] != 0U) {
                read_count[pidx]--;
            }
            maybe_free(pidx);
        }

        void retire_write(uint32_t pidx) {
            if (write_count[pidx] != 0U) {
                write_count[pidx]--;
            }
            maybe_free(pidx);
        }

        TInst rename(const TInstRaw &raw) {
            TInst out;
            uint32_t srcs[8];
            uint32_t src_count = 0;

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
            out.pmask = raw.mask;
            for (int i = 0; i < 4; ++i) {
                out.rd_pidx[i] = 0;
                out.rs1_pidx[i] = 0;
                out.rs2_pidx[i] = 0;
            }

            if (raw.rs1_type == TOPRandType::TREG) {
                uint32_t cnt = ((uint32_t(raw.rs1_datatype) >> 4U) == 0U) ? 1U :
                               (((uint32_t(raw.rs1_datatype) >> 4U) == 1U) ? 2U : 4U);
                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t pidx = map[raw.rs1_vidx.template to<uint32_t>() + i];
                    out.rs1_pidx[i] = pidx;
                    srcs[src_count++] = pidx;
                }
            } else if (raw.rs1_type == TOPRandType::TMASK) {
                uint32_t midx = raw.rs1_vidx.template to<uint32_t>() & 0x3U;
                if (midx != 0U) {
                    out.rs1_pidx[0] = midx;
                }
            }

            if (raw.rs2_type == TOPRandType::TREG) {
                uint32_t cnt = ((uint32_t(raw.rs2_datatype) >> 4U) == 0U) ? 1U :
                               (((uint32_t(raw.rs2_datatype) >> 4U) == 1U) ? 2U : 4U);
                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t pidx = map[raw.rs2_vidx.template to<uint32_t>() + i];
                    out.rs2_pidx[i] = pidx;
                    srcs[src_count++] = pidx;
                }
            } else if (raw.rs2_type == TOPRandType::TMASK) {
                uint32_t midx = raw.rs2_vidx.template to<uint32_t>() & 0x3U;
                if (midx != 0U) {
                    out.rs2_pidx[0] = midx;
                }
            }

            if (raw.rd_type == TOPRandType::TREG) {
                uint32_t cnt = ((uint32_t(raw.rd_datatype) >> 4U) == 0U) ? 1U :
                               (((uint32_t(raw.rd_datatype) >> 4U) == 1U) ? 2U : 4U);
                uint32_t rd_vidx[4];
                uint32_t rd_old_pidx[4];
                bool rd_need_alloc[4];
                for (uint32_t i = 0; i < cnt; ++i) {
                    rd_vidx[i] = raw.rd_vidx.template to<uint32_t>() + i;
                    rd_old_pidx[i] = map[rd_vidx[i]];
                    rd_need_alloc[i] = read_count[rd_old_pidx[i]] != 0U ||
                                       source_used(srcs, src_count, rd_old_pidx[i]);
                }

                for (uint32_t i = 0; i < src_count; ++i) {
                    read_count[srcs[i]]++;
                }

                for (uint32_t i = 0; i < cnt; ++i) {
                    uint32_t pidx = rd_old_pidx[i];
                    if (rd_need_alloc[i]) {
                        pidx = alloc();
                        map[rd_vidx[i]] = pidx;
                        mapped[rd_old_pidx[i]] = false;
                        maybe_free(rd_old_pidx[i]);
                        mapped[pidx] = true;
                    }
                    out.rd_pidx[i] = pidx;
                    write_count[pidx]++;
                }
            } else if (raw.rd_type == TOPRandType::TMASK) {
                for (uint32_t i = 0; i < src_count; ++i) {
                    read_count[srcs[i]]++;
                }
                uint32_t midx = raw.rd_vidx.template to<uint32_t>() & 0x3U;
                if (midx != 0U) {
                    out.rd_pidx[0] = midx;
                }
            } else {
                for (uint32_t i = 0; i < src_count; ++i) {
                    read_count[srcs[i]]++;
                }
            }

            return out;
        }
    };

    auto vt_base = [&](uint32_t &rng, uint32_t width) {
        uint32_t limit = VTREG_SIZE - width;
        return rng_next(rng) % (limit + 1U);
    };

    auto make_inst = [&](uint32_t &rng, uint32_t n) {
        TInstRaw inst = clear_raw();
        uint32_t pattern = rng_next(rng) % 10U;
        DType rd_dtype = dtype_for_group(rng_next(rng) % 3U);
        DType rs1_dtype = dtype_for_group(rng_next(rng) % 3U);
        DType rs2_dtype = dtype_for_group(rng_next(rng) % 3U);
        inst.rd_datatype = rd_dtype;
        inst.rs1_datatype = rs1_dtype;
        inst.rs2_datatype = rs2_dtype;
        inst.mask = rng_next(rng) % 4U;

        if (pattern <= 3U) {
            inst.rd_type = TOPRandType::TREG;
            inst.rs1_type = TOPRandType::TREG;
            inst.rs2_type = TOPRandType::TREG;
            inst.rd_vidx = vt_base(rng, reg_count(rd_dtype));
            inst.rs1_vidx = vt_base(rng, reg_count(rs1_dtype));
            inst.rs2_vidx = vt_base(rng, reg_count(rs2_dtype));
            if ((n % 13U) == 0U) {
                inst.rd_vidx = inst.rs1_vidx;
                inst.rd_datatype = inst.rs1_datatype;
            }
        } else if (pattern == 4U) {
            inst.rd_type = TOPRandType::TREG;
            inst.rs1_type = TOPRandType::TTEMP;
            inst.rd_vidx = vt_base(rng, reg_count(rd_dtype));
        } else if (pattern == 5U) {
            inst.rd_type = TOPRandType::TREG;
            inst.rs1_type = TOPRandType::TACC;
            inst.rd_vidx = vt_base(rng, reg_count(rd_dtype));
        } else if (pattern == 6U) {
            inst.rd_type = TOPRandType::TTEMP;
            inst.rs1_type = TOPRandType::TREG;
            inst.rs1_vidx = vt_base(rng, reg_count(rs1_dtype));
        } else if (pattern == 7U) {
            inst.rd_type = TOPRandType::TACC;
            inst.rs1_type = TOPRandType::TREG;
            inst.rs1_vidx = vt_base(rng, reg_count(rs1_dtype));
        } else if (pattern == 8U) {
            inst.rd_type = TOPRandType::TMASK;
            inst.rs1_type = TOPRandType::TMASK;
            inst.rs2_type = TOPRandType::TREG;
            inst.rd_vidx = 1U + (rng_next(rng) % 3U);
            inst.rs1_vidx = 1U + (rng_next(rng) % 3U);
            inst.rs2_vidx = vt_base(rng, reg_count(rs2_dtype));
        } else {
            inst.rd_type = TOPRandType::TREG;
            inst.rs1_type = TOPRandType::INVALID;
            inst.rs2_type = TOPRandType::TMASK;
            inst.rd_vidx = vt_base(rng, reg_count(rd_dtype));
            inst.rs2_vidx = 1U + (rng_next(rng) % 3U);
        }
        return inst;
    };

    auto collect_sources = [&](const TInstRaw &raw, const TInst &renamed, RenameRetireRead &rr) {
        for (int i = 0; i < 8; ++i) {
            rr.pidx[i] = 0;
        }
        rr.valid = 0;
        uint32_t slot = 0;
        if (raw.rs1_type == TOPRandType::TREG) {
            uint32_t cnt = reg_count(raw.rs1_datatype);
            for (uint32_t i = 0; i < cnt; ++i) {
                rr.pidx[slot] = renamed.rs1_pidx[i];
                rr.valid = rr.valid | Int<8>(1U << slot);
                slot++;
            }
        }
        if (raw.rs2_type == TOPRandType::TREG) {
            uint32_t cnt = reg_count(raw.rs2_datatype);
            for (uint32_t i = 0; i < cnt; ++i) {
                rr.pidx[slot] = renamed.rs2_pidx[i];
                rr.valid = rr.valid | Int<8>(1U << slot);
                slot++;
            }
        }
    };

    auto collect_dests = [&](const TInstRaw &raw, const TInst &renamed, RenameRetireWrite &rw) {
        for (int i = 0; i < 4; ++i) {
            rw.pidx[i] = 0;
        }
        rw.valid = 0;
        if (raw.rd_type == TOPRandType::TREG) {
            uint32_t cnt = reg_count(raw.rd_datatype);
            for (uint32_t i = 0; i < cnt; ++i) {
                rw.pidx[i] = renamed.rd_pidx[i];
                rw.valid = rw.valid | Int<4>(1U << i);
            }
        }
    };

    auto same_output = [](const TInst &a, const TInst &b) {
        if (a.subopcode != b.subopcode || a.args != b.args || a.funct5 != b.funct5 ||
            a.mask != b.mask || a.rd_type != b.rd_type || a.rs1_type != b.rs1_type ||
            a.rs2_type != b.rs2_type || a.rd_vidx != b.rd_vidx ||
            a.rs1_vidx != b.rs1_vidx || a.rs2_vidx != b.rs2_vidx ||
            a.pmask != b.pmask) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (a.rd_pidx[i] != b.rd_pidx[i] ||
                a.rs1_pidx[i] != b.rs1_pidx[i] ||
                a.rs2_pidx[i] != b.rs2_pidx[i]) {
                return false;
            }
        }
        return true;
    };

    constexpr uint32_t INST_COUNT = 5000;
    constexpr uint32_t PENDING = 32768;
    std::array<TInstRaw, INST_COUNT> insts;
    std::array<PendingRetire, PENDING> pending;
    uint32_t rng = 0x12345678U;
    uint32_t cycle_inst = 0;

    for (uint32_t i = 0; i < INST_COUNT; ++i) {
        insts[i] = make_inst(rng, i);
    }
    for (uint32_t i = 0; i < PENDING; ++i) {
        pending[i].valid = false;
        pending[i].is_write = false;
        pending[i].ready_inst = 0;
        pending[i].pidx = 0;
    }

    auto add_pending = [&](bool is_write, uint32_t pidx, uint32_t ready_inst) {
        for (uint32_t i = 0; i < PENDING; ++i) {
            if (!pending[i].valid) {
                pending[i].valid = true;
                pending[i].is_write = is_write;
                pending[i].ready_inst = ready_inst;
                pending[i].pidx = pidx;
                return;
            }
        }
        fail("pending retire table overflow");
    };

    auto drain_due = [&](Model &model, uint32_t upto_inst) {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            RenameRetireRead rr;
            RenameRetireWrite rw;
            for (int i = 0; i < 8; ++i) {
                rr.pidx[i] = 0;
            }
            for (int i = 0; i < 4; ++i) {
                rw.pidx[i] = 0;
            }
            rr.valid = 0;
            rw.valid = 0;
            uint32_t rc = 0;
            uint32_t wc = 0;
            for (uint32_t i = 0; i < PENDING; ++i) {
                if (pending[i].valid && pending[i].ready_inst <= upto_inst) {
                    if (!pending[i].is_write && rc < 8U) {
                        rr.pidx[rc] = pending[i].pidx;
                        rr.valid = rr.valid | Int<8>(1U << rc);
                        pending[i].valid = false;
                        rc++;
                        progressed = true;
                    } else if (pending[i].is_write && wc < 4U) {
                        rw.pidx[wc] = pending[i].pidx;
                        rw.valid = rw.valid | Int<4>(1U << wc);
                        pending[i].valid = false;
                        wc++;
                        progressed = true;
                    }
                }
            }
            if (progressed) {
                for (uint32_t i = 0; i < rc; ++i) {
                    model.retire_read(rr.pidx[i].template to<uint32_t>());
                }
                for (uint32_t i = 0; i < wc; ++i) {
                    model.retire_write(rw.pidx[i].template to<uint32_t>());
                }
                if (rr.valid != Int<8>(0) && !retireRead(rr)) {
                    fail("stress retireRead rejected");
                }
                if (rw.valid != Int<4>(0) && !retireWrite(rw)) {
                    fail("stress retireWrite rejected");
                }
                step();
                step();
            }
        }
    };

    sim_reset();
    Model model;
    model.reset();

    for (uint32_t n = 0; n < INST_COUNT; ++n) {
        cycle_inst = n;
        drain_due(model, cycle_inst);

        TInst expected = model.rename(insts[n]);
        feed_inst = insts[n];
        feed_valid = true;
        step();

        TInst actual;
        if (!s2GetResult(actual)) {
            fail("stress result not ready");
        }
        if (!same_output(actual, expected)) {
            std::printf("mismatch n=%u rd_type=%u rs1_type=%u rs2_type=%u rd_vidx=%u rs1_vidx=%u rs2_vidx=%u rd_dtype=%u rs1_dtype=%u rs2_dtype=%u\n",
                        n,
                        uint32_t(insts[n].rd_type),
                        uint32_t(insts[n].rs1_type),
                        uint32_t(insts[n].rs2_type),
                        insts[n].rd_vidx.template to<uint32_t>(),
                        insts[n].rs1_vidx.template to<uint32_t>(),
                        insts[n].rs2_vidx.template to<uint32_t>(),
                        uint32_t(insts[n].rd_datatype),
                        uint32_t(insts[n].rs1_datatype),
                        uint32_t(insts[n].rs2_datatype));
            for (int i = 0; i < 4; ++i) {
                std::printf("slot %d exp rd=%u rs1=%u rs2=%u got rd=%u rs1=%u rs2=%u\n",
                            i,
                            expected.rd_pidx[i].template to<uint32_t>(),
                            expected.rs1_pidx[i].template to<uint32_t>(),
                            expected.rs2_pidx[i].template to<uint32_t>(),
                            actual.rd_pidx[i].template to<uint32_t>(),
                            actual.rs1_pidx[i].template to<uint32_t>(),
                            actual.rs2_pidx[i].template to<uint32_t>());
            }
            fail("stress rename output mismatch");
        }

        RenameRetireRead rr;
        RenameRetireWrite rw;
        collect_sources(insts[n], actual, rr);
        collect_dests(insts[n], actual, rw);
        uint32_t read_delay = 1U + (rng_next(rng) % 37U);
        uint32_t write_delay = 1U + (rng_next(rng) % 53U);
        for (int i = 0; i < 8; ++i) {
            if (rr.valid.pick(i)) {
                add_pending(false, rr.pidx[i].template to<uint32_t>(), n + read_delay + uint32_t(i % 3));
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (rw.valid.pick(i)) {
                add_pending(true, rw.pidx[i].template to<uint32_t>(), n + write_delay + uint32_t(i % 2));
            }
        }

        step();
    }

    drain_due(model, INST_COUNT + 128U);
    std::printf("rename stress passed\n");
}
