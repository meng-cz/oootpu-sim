#include <cstdio>
#include <cstdlib>

#include <defhelper.hpp>
#include <run.hpp>

#include "../../src/header/rename.hpp"

TOP("../../src/core/Rename.hpp");
PROJECT("../../src");

REQUEST_READY(s2GetResult, RESP(TInst) inst);
REQUEST_READY(retireRead, ARG(RenameRetireRead) info);

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
        std::printf("rename smoke failed: %s\n", msg);
        std::exit(1);
    };

    auto step = [&]() {
        sim_execute();
        sim_commit();
    };

    sim_reset();

    feed_inst.subopcode = 0x70;
    feed_inst.args = 0;
    feed_inst.funct5 = 0;
    feed_inst.mask = 1;
    feed_inst.rd_datatype = DType::DTYPE_SINT16;
    feed_inst.rs1_datatype = DType::DTYPE_SINT16;
    feed_inst.rs2_datatype = DType::DTYPE_SINT32;
    feed_inst.rd_type = TOPRandType::TREG;
    feed_inst.rs1_type = TOPRandType::TREG;
    feed_inst.rs2_type = TOPRandType::TREG;
    feed_inst.rd_vidx = 1;
    feed_inst.rs1_vidx = 1;
    feed_inst.rs2_vidx = 8;
    feed_valid = true;

    step();

    TInst out;
    if (!s2GetResult(out)) {
        fail("s2 result not ready");
    }
    if (out.rs1_pidx[0] != 1 || out.rs1_pidx[1] != 2 ||
        out.rs2_pidx[0] != 8 || out.rs2_pidx[3] != 11 ||
        out.rd_pidx[0] != VTREG_SIZE || out.rd_pidx[1] != VTREG_SIZE + 1 ||
        out.pmask != 1) {
        fail("unexpected physical indices");
    }

    step();

    sim_reset();

    feed_inst.subopcode = 0x70;
    feed_inst.args = 0;
    feed_inst.funct5 = 0;
    feed_inst.mask = 0;
    feed_inst.rd_datatype = DType::DTYPE_SINT8;
    feed_inst.rs1_datatype = DType::DTYPE_SINT8;
    feed_inst.rs2_datatype = DType::DTYPE_SINT8;
    feed_inst.rd_type = TOPRandType::TREG;
    feed_inst.rs1_type = TOPRandType::INVALID;
    feed_inst.rs2_type = TOPRandType::INVALID;
    feed_inst.rd_vidx = 20;
    feed_inst.rs1_vidx = 0;
    feed_inst.rs2_vidx = 0;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("in-place first write result not ready");
    }
    if (out.rd_pidx[0] != 20) {
        fail("rd without in-flight read dependency did not rename in place");
    }
    step();

    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("in-place second write result not ready");
    }
    if (out.rd_pidx[0] != 20) {
        fail("consecutive rd without read dependency did not stay in place");
    }
    step();

    feed_inst.rd_vidx = 21;
    feed_inst.rs1_vidx = 21;
    feed_inst.rs1_type = TOPRandType::TREG;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("read-dependent write result not ready");
    }
    if (out.rs1_pidx[0] != 21 || out.rd_pidx[0] != VTREG_SIZE) {
        fail("rd with same-instruction read dependency did not allocate new physical register");
    }
    step();

    feed_inst.rs1_type = TOPRandType::INVALID;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("post-allocation in-place result not ready");
    }
    if (out.rd_pidx[0] != VTREG_SIZE) {
        fail("mapped rd without read dependency did not reuse current physical register");
    }
    step();

    sim_reset();

    feed_inst.subopcode = 0x70;
    feed_inst.args = 0;
    feed_inst.funct5 = 0;
    feed_inst.mask = 0;
    feed_inst.rd_datatype = DType::DTYPE_SINT8;
    feed_inst.rs1_datatype = DType::DTYPE_SINT8;
    feed_inst.rs2_datatype = DType::DTYPE_SINT8;
    feed_inst.rd_type = TOPRandType::TREG;
    feed_inst.rs1_type = TOPRandType::TREG;
    feed_inst.rs2_type = TOPRandType::INVALID;
    feed_inst.rd_vidx = 1;
    feed_inst.rs1_vidx = 1;
    feed_inst.rs2_vidx = 0;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("first retire setup result not ready");
    }
    if (out.rs1_pidx[0] != 1 || out.rd_pidx[0] != VTREG_SIZE) {
        fail("first retire setup rename mismatch");
    }
    step();

    RenameRetireRead rr;
    for (int i = 0; i < 8; ++i) {
        rr.pidx[i] = 0;
    }
    rr.pidx[0] = 1;
    rr.valid = 0x01;
    if (!retireRead(rr)) {
        fail("retireRead rejected");
    }

    feed_inst.rd_vidx = 2;
    feed_inst.rs1_vidx = 2;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("second retire setup result not ready");
    }
    if (out.rs1_pidx[0] != 2 || out.rd_pidx[0] != VTREG_SIZE + 1) {
        fail("retire was incorrectly visible to same-cycle rename");
    }
    step();

    feed_inst.rd_vidx = 3;
    feed_inst.rs1_vidx = 3;
    feed_valid = true;
    step();

    if (!s2GetResult(out)) {
        fail("third retire setup result not ready");
    }
    if (out.rs1_pidx[0] != 3 || out.rd_pidx[0] != VTREG_SIZE + 2) {
        fail("buffered retire was not merged before later rename");
    }

    step();

    sim_reset();

    feed_inst.subopcode = 0x70;
    feed_inst.args = 0;
    feed_inst.funct5 = 0;
    feed_inst.mask = 0;
    feed_inst.rd_datatype = DType::DTYPE_SINT8;
    feed_inst.rs1_datatype = DType::DTYPE_SINT8;
    feed_inst.rs2_datatype = DType::DTYPE_SINT8;
    feed_inst.rd_type = TOPRandType::TREG;
    feed_inst.rs1_type = TOPRandType::TREG;
    feed_inst.rs2_type = TOPRandType::INVALID;
    feed_inst.rd_vidx = 1;
    feed_inst.rs1_vidx = 1;
    feed_inst.rs2_vidx = 0;

    for (int i = 0; i < PTREG_SIZE - VTREG_SIZE; ++i) {
        feed_valid = true;
        step();
        if (!s2GetResult(out)) {
            fail("free-list drain result not ready");
        }
        if (out.rd_pidx[0] != VTREG_SIZE + i) {
            fail("free-list drain allocation mismatch");
        }
        step();
    }

    feed_valid = true;
    step();
    if (s2GetResult(out)) {
        fail("exhausted rename unexpectedly produced a result");
    }

    for (int i = 0; i < 8; ++i) {
        rr.pidx[i] = 0;
    }
    rr.pidx[0] = 1;
    rr.valid = 0x01;
    if (!retireRead(rr)) {
        fail("exhaustion retireRead rejected");
    }
    step();
    if (s2GetResult(out)) {
        fail("same-cycle retire incorrectly unblocked held rename");
    }

    step();
    if (s2GetResult(out)) {
        fail("buffered retire incorrectly unblocked before committed count state");
    }

    step();
    if (!s2GetResult(out)) {
        fail("held rename did not retry after resource became free");
    }
    if (out.rd_pidx[0] != 1) {
        fail("held rename did not reuse retired physical register");
    }

    step();
    std::printf("rename smoke passed\n");
}
