#pragma once

#include <cassert>
#include <defhelper.hpp>

#include "../header/alu.hpp"
#include "../header/gemm.hpp"
#include "ALU.hpp"
#include "../gemm/GEMMUnit.hpp"

// Simulation-only integration wrapper for the ALU TACCGET path and GEMMUnit.

REQUEST_READY(aluWriteback, ARG(AluWriteback) wb);
REQUEST_READY(retireInst, ARG(AluRetireInfo) info);
REQUEST(retirePAcc, ARG(uint8_t) pacc);

CHILD_INSTANCE(ALU, alu);
CHILD_INSTANCE(GEMMUnit, gemm);

USE_CHILD_SERVICE_PORT(alu, s0AluInstFire, child_s0AluInstFire, ARG(AluInstMeta) inst);
USE_CHILD_SERVICE_PORT(alu, sNAluMemInput, child_sNAluMemInput, ARG(AluMemInput) in);
USE_CHILD_SERVICE_PORT(alu, s0AccgetStart, child_s0AccgetStart, ARG(AluAccgetMeta) meta);

USE_CHILD_SERVICE_PORT(gemm, s0FireGemmInst, child_s0FireGemmInst,
                       ARG(uint8_t) pacc, ARG(DType) dtype, ARG(bool) transA, ARG(bool) transB);
USE_CHILD_SERVICE_PORT(gemm, s1s2MemInput, child_s1s2MemInput, ARG(bool) selB, ARG(Tile) data);

SERVICE_READY(s0AluInstFire, true, ARG(AluInstMeta) inst) {
    bool ok = child_s0AluInstFire(inst);
    assert(ok);
}

SERVICE(sNAluMemInput, ARG(AluMemInput) in) {
    child_sNAluMemInput(in);
}

SERVICE_READY(s0AccgetStart, true, ARG(AluAccgetMeta) meta) {
    bool ok = child_s0AccgetStart(meta);
    assert(ok);
}

SERVICE_READY(s0FireGemmInst, true, ARG(uint8_t) pacc, ARG(DType) dtype, ARG(bool) transA, ARG(bool) transB) {
    bool ok = child_s0FireGemmInst(pacc, dtype, transA, transB);
    assert(ok);
}

SERVICE(s1s2MemInput, ARG(bool) selB, ARG(Tile) data) {
    child_s1s2MemInput(selB, data);
}

CONNECT_CR_CS(alu, s1FireAccget, gemm, s0FireAccget);
CONNECT_CR_CS(alu, sNAccOutput, gemm, sNAccInput);
CONNECT_CR_CS(gemm, accOutput, alu, accTempInput);
CONNECT_CR_CS(gemm, retirePTemp, alu, accgetRetirePTemp);

CONNECT_CR_R(alu, aluWriteback, aluWriteback);
CONNECT_CR_R(alu, retireInst, retireInst);
CONNECT_CR_R(gemm, retirePAcc, retirePAcc);

USE_CHILD_QUERY(gemm, gemmFeedingReady, child_gemmFeedingReady, bool);
USE_CHILD_QUERY(gemm, accgetReady, child_accgetReady, bool);

QUERY(gemmFeedingReady, bool) {
    return child_gemmFeedingReady();
}

QUERY(gemmAccgetReady, bool) {
    return child_accgetReady();
}
