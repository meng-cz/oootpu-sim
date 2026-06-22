#include <cstdio>
#include <cstdlib>

#include <defhelper.hpp>
#include <run.hpp>

#include "../header.hpp"

TOP("../Top.hpp");
PROJECT("..");

REQUEST(step);
QUERY(status, Clog2Status);

SIMULATION() {
    auto fail = [&](const char *msg, uint32_t got, uint32_t expected) {
        std::printf("clog2demo failed: %s got=%u expected=%u\n", msg, got, expected);
        std::exit(1);
    };
    auto check = [&](bool cond, const char *msg, uint32_t got, uint32_t expected) {
        if (!cond) fail(msg, got, expected);
    };

    check(WIDTH_A == 4, "WIDTH_A", WIDTH_A, 4);
    check(WIDTH_B == 4, "WIDTH_B", WIDTH_B, 4);
    check(WIDTH_SUM == 5, "WIDTH_SUM", WIDTH_SUM, 5);
    Clog2Status snap = status();
    check(snap.width_a == 4U, "query width_a", snap.width_a, 4);
    check(snap.width_b == 4U, "query width_b", snap.width_b, 4);
    check(snap.width_sum == 5U, "query width_sum", snap.width_sum, 5);
    check(snap.idx_a.to<uint32_t>() == 0U, "initial idx_a", snap.idx_a.to<uint32_t>(), 0);
    check(snap.idx_b.to<uint32_t>() == 0U, "initial idx_b", snap.idx_b.to<uint32_t>(), 0);

    for (uint32_t cycle = 0; cycle < 9; ++cycle) {
        step();
        sim_execute();
        sim_commit();
    }

    snap = status();
    check(snap.idx_a.to<uint32_t>() == 0U, "wrapped idx_a", snap.idx_a.to<uint32_t>(), 0);
    check(snap.idx_b.to<uint32_t>() == 9U, "advanced idx_b", snap.idx_b.to<uint32_t>(), 9);
    check(snap.a_wraps == false, "post-wrap a_wraps", snap.a_wraps ? 1U : 0U, 0);
    check(snap.b_wraps == false, "post-wrap b_wraps", snap.b_wraps ? 1U : 0U, 0);

    step();
    sim_execute();
    sim_commit();

    snap = status();
    check(snap.idx_a.to<uint32_t>() == 1U, "next idx_a", snap.idx_a.to<uint32_t>(), 1);
    check(snap.idx_b.to<uint32_t>() == 10U, "next idx_b", snap.idx_b.to<uint32_t>(), 10);

    std::printf("clog2demo passed\n");
}
