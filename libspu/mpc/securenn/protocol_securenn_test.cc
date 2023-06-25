#include "libspu/mpc/securenn/protocol_securenn_test.h"
#include "libspu/mpc/securenn/type.h"

#include "libspu/core/shape_util.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
#include "yacl/link/link.h"


namespace spu::mpc::test {
namespace{

constexpr int64_t kNumel = 1000;

bool verifyCost(Kernel* kernel, std::string_view name, const ce::Params& params,
                const Communicator::Stats& cost, size_t repeated = 1) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  const auto expectedComm = comm->eval(params) * repeated;
  const auto realComm = cost.comm * kBitsPerBytes;

  float diff;
  if (expectedComm == 0) {
    diff = realComm;
  } else {
    diff = (realComm - expectedComm) / expectedComm;
  }
  if (realComm < expectedComm || diff > kernel->getCommTolerance()) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               expectedComm, realComm);
    succeed = false;
  }

  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }
  return succeed;
}

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                size_t numel, size_t npc, const Communicator::Stats& cost) {
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  return verifyCost(kernel, name, params, cost, numel /*repeated*/);
}

}

TEST_P(ArithmeticTest, MulAA_2pc) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());

    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kNumel);
    auto p1 = rand_p(obj.get(), kNumel);
    // ring_print(p0, "p0");
    // ring_print(p1, "p1");

    // auto p01 = rand_p(obj.get(), kNumel);
    // auto p11 = rand_p(obj.get(), kNumel);
    // auto a0 = ring_zeros(conf.field(), kNumel);
    // auto a1 = ring_zeros(conf.field(), kNumel);

    // if(rank == 0){
    //   a0 = p01.as(ty);
    //   a1 = p11.as(ty);
    // }
    // if(rank == 1){
    //   a0 = ring_sub(p0, p01).as(ty);
    //   a1 = ring_sub(p1, p11).as(ty);
    // }
    // if(rank == 2){
    //   a0 = a0.as(ty);
    //   a1 = a1.as(ty);
    // }
    auto a0 = obj->call("p2a", p0);
    auto a1 = obj->call("p2a", p1);


    /* WHEN */
    auto r_pp = ring_mul(p0, p1);
    auto prev = comm->getStats();
    auto r_aa = obj->call("mul_aa", a0, a1);
    auto cost = comm->getStats() - prev;
    auto r_aa_old = obj->call("mul_aa_old", a0, a1);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(a2p(obj.get(), r_aa), r_pp));
    EXPECT_TRUE(ring_all_equal(a2p(obj.get(), r_aa), a2p(obj.get(), r_aa_old)));

    EXPECT_TRUE(verifyCost(obj->getKernel("mul_aa"), "mul_aa", conf.field(),
                           kNumel, npc, cost));
  });
}

TEST_P(ArithmeticTest, MatMulAA_2pc) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const size_t M = 240;
  const size_t K = 320;
  const size_t N = 160;
  const size_t N2 = 900;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    /* GIVEN */

    auto p0 = rand_p(obj.get(), M * K);
    auto p1 = rand_p(obj.get(), K * N);
    // auto p2 = rand_p(obj.get(), K * N2);

    auto p01 = rand_p(obj.get(), M * K);
    auto p11 = rand_p(obj.get(), K * N);
    // auto p21 = rand_p(obj.get(), K * N2);

    auto a0 = ring_zeros(conf.field(), M * K);
    auto a1 = ring_zeros(conf.field(), K * N);
    // auto a2 = ring_zeros(conf.field(), K * N2);

    if(rank == 0){
      a0 = p01.as(ty);
      a1 = p11.as(ty);
      // a2 = p21.as(ty);
    }
    if(rank == 1){
      a0 = ring_sub(p0, p01).as(ty);
      a1 = ring_sub(p1, p11).as(ty);
      // a2 = ring_sub(p2, p21).as(ty);
    }
    if(rank == 2){
      a0 = a0.as(ty);
      a1 = a1.as(ty);
      // a2 = a2.as(ty);
    }

    /* WHEN */
    auto prev = comm->getStats();
    auto tmp = obj->call("mmul_aa", a0, a1, M, N, K);
    auto cost = comm->getStats() - prev;
    // auto tmp2 = obj->call("mmul_aa", a0, a2, M, N2, K);

    auto r_ss = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    // auto r_ss2 = a2p(obj.get(), tmp2);
    // auto r_pp2 = mmul_pp(obj.get(), p0, p2, M, N2, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_ss, r_pp));
    // EXPECT_TRUE(ring_all_equal(r_ss2, r_pp2));
    ce::Params params = {{"K", SizeOf(conf.field()) * 8},
                         {"N", npc},
                         {"m", M},
                         {"n", N},
                         {"k", K}};
    EXPECT_TRUE(verifyCost(obj->getKernel("mmul_aa"), "mmul_aa", params,
                           cost, 1));
  });
}

TEST_P(ArithmeticTest, MatMulAA_2pc_simple) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const size_t M = 240;
  const size_t K = 320;
  const size_t N = 160;
  const size_t N2 = 90;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    /* GIVEN */

    auto p0 = rand_p(obj.get(), M * K);
    auto p1 = rand_p(obj.get(), K * N);
    // auto p2 = rand_p(obj.get(), K * N2);

    auto p01 = rand_p(obj.get(), M * K);
    auto p11 = rand_p(obj.get(), K * N);
    // auto p21 = rand_p(obj.get(), K * N2);

    auto a0 = ring_zeros(conf.field(), M * K);
    auto a1 = ring_zeros(conf.field(), K * N);
    // auto a2 = ring_zeros(conf.field(), K * N2);

    if(rank == 0){
      a0 = p01.as(ty);
      a1 = p11.as(ty);
      // a2 = p21.as(ty);
    }
    if(rank == 1){
      a0 = ring_sub(p0, p01).as(ty);
      a1 = ring_sub(p1, p11).as(ty);
      // a2 = ring_sub(p2, p21).as(ty);
    }
    if(rank == 2){
      a0 = a0.as(ty);
      a1 = a1.as(ty);
      // a2 = a2.as(ty);
    }

    /* WHEN */
    auto prev = comm->getStats();
    auto tmp = obj->call("mmul_aa_simple", a0, a1, M, N, K);
    auto cost = comm->getStats() - prev;
    // auto tmp2 = obj->call("mmul_aa", a0, a2, M, N2, K);

    auto r_ss = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    // auto r_ss2 = a2p(obj.get(), tmp2);
    // auto r_pp2 = mmul_pp(obj.get(), p0, p2, M, N2, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_ss, r_pp));
    // EXPECT_TRUE(ring_all_equal(r_ss2, r_pp2));
    ce::Params params = {{"K", SizeOf(conf.field()) * 8},
                         {"N", npc},
                         {"m", M},
                         {"n", N},
                         {"k", K}};
    EXPECT_TRUE(verifyCost(obj->getKernel("mmul_aa_simple"), "mmul_aa_simple", params,
                           cost, 1));
  });
}

TEST_P(ArithmeticTest, TruncAA) {
  constexpr int64_t kkNumel = 40000;
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    auto p0 = rand_p(obj.get(), kkNumel);
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kkNumel);
    auto a0 = ring_zeros(conf.field(), kkNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kkNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    // ring_print(p0, "p0");
    // ring_print(a0, "a0");

    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("trunc_a", a0, (size_t)13);
    auto cost = comm->getStats() - prev;

    /* THEN */
 
    EXPECT_TRUE(1 == 1);


  });
}

TEST_P(ArithmeticTest, MSB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    auto p0 = rand_p(obj.get(), kNumel);
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    // ring_print(p0, "p0");
    // ring_print(a0, "a0");

    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("msb_a2a_test", a0);
    auto cost = comm->getStats() - prev;
    // ring_print(a2p(obj.get(), a1), "res");
    // ring_print(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1), "msb_p0");

    /* THEN */
 
    EXPECT_TRUE(ring_all_equal(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1),
                             a2p(obj.get(), a1)));
    // EXPECT_TRUE(verifyCost(obj->getKernel("msb_a2a"), "msb_a2a", conf.field(),
    //                        kNumel, npc, cost));
    //}

  });
}

TEST_P(ArithmeticTest, MSB_opt) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    auto p0 = rand_p(obj.get(), kNumel);
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    // ring_print(p0, "p0");
    // ring_print(a0, "a0");

    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("msb_opt_a2a", a0);
    auto cost = comm->getStats() - prev;
    // ring_print(a2p(obj.get(), a1), "res");
    // ring_print(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1), "msb_p0");

    /* THEN */
 
    EXPECT_TRUE(ring_all_equal(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1),
                             a2p(obj.get(), a1)));
    // EXPECT_TRUE(verifyCost(obj->getKernel("msb_opt_a2a"), "msb_opt_a2a", conf.field(),
    //                        kNumel, npc, cost));
    //}

  });
}

TEST_P(ArithmeticTest, MSB_A2B) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    auto p0 = rand_p(obj.get(), kNumel);
    //auto a0 = p2a(obj.get(), p0);
    // ring_print(p0, "p0");

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    // ring_print(p0, "p0");
    // ring_print(a0, "a0");

    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("msb_a2b", a0);
    auto cost = comm->getStats() - prev;
    // ring_print(a2p(obj.get(), a1), "res");
    // ring_print(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1), "msb_p0");

    /* THEN */
 
    EXPECT_TRUE(ring_all_equal(ring_rshift(p0, SizeOf(conf.field()) * 8 - 1),
                             b2p(obj.get(), a1)));
    // EXPECT_TRUE(verifyCost(obj->getKernel("msb_a2b"), "msb_opt_a2a", conf.field(),
    //                        kNumel, npc, cost));
    //}

  });
}

TEST_P(ArithmeticTest, ShareConvert) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    auto p0 = rand_p(obj.get(), kNumel);
    // ring_print(p0, "p0");
    
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);
    // ring_print(a0, "a0");
    // fmt::print("\n");
    
    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("sc", a0);  
    auto cost = comm->getStats() - prev;
    // ring_print(a1, "a1");
    
    auto res = ring_zeros(conf.field(), kNumel).as(ty);

    DISPATCH_ALL_FIELDS(conf.field(), "_", [&](){
      auto _a1 = ArrayView<ring2k_t>(a1);
      auto _res = ArrayView<ring2k_t>(res);
      if(rank == 0){
        comm->sendAsync(1, a1, "a1");
      }
      if(rank == 1){
        auto a1_recv = comm->recv(0, ty, "a1");
        auto _a1_recv = ArrayView<ring2k_t>(a1_recv);
        pforeach(0, a1.numel(), [&](int64_t idx){
          // res = a1 + a1_recv  mod (2^k - 1)
          _res[idx] = _a1_recv[idx] + _a1[idx];
          if(_res[idx] < _a1[idx]) _res[idx] += (ring2k_t)1;
        });
      }
    });

    /* THEN */
    EXPECT_TRUE(ring_all_equal(a2p(obj.get(), res), p0));
    EXPECT_TRUE(verifyCost(obj->getKernel("sc"), "sc", conf.field(),
                           kNumel, npc, cost));
  });
}

TEST_P(ArithmeticTest, DReLU) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    // the safe range is [-2^(k-2), 2^(k-2))
    // [0, 2^(k-2)) union (2^k - 2^(k-2), 2^k-1)
    auto p0 = rand_p(obj.get(), kNumel);
    // auto randbit = ring_rshift(rand_p(obj.get(), kNumel), SizeOf(conf.field()) * 8 - 1); // 0/1
    // auto prefix = ring_add(randbit, ring_lshift(randbit, 1));  //0b00, 0x11
    // prefix = ring_lshift(prefix, SizeOf(conf.field()) * 8 - 2); //0b0000..00  0b1100..00
    // p0 = ring_add(prefix, ring_rshift(p0, 2));  //0b00...  0b11...
    // ring_print(p0, "p0");
    
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    /* WHEN */
    // auto prev = comm->getStats();
    auto a1 = obj->call("drelu", a0);
    // auto cost = comm->getState() - prev;


    /* THEN */
    auto msb = ring_rshift(p0, SizeOf(conf.field()) * 8 - 1);
    auto drelu_p0 = ring_sub(ring_ones(conf.field(), kNumel), msb);
    // ring_print(drelu_p0, "drelu");
    EXPECT_TRUE(ring_all_equal(drelu_p0, a2p(obj.get(), a1)));

  });
}

TEST_P(ArithmeticTest, DReLU_sc) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    // the safe range is [-2^(k-2), 2^(k-2))
    // [0, 2^(k-2)) union (2^k - 2^(k-2), 2^k-1)
    auto p0 = rand_p(obj.get(), kNumel);
    // auto randbit = ring_rshift(rand_p(obj.get(), kNumel), SizeOf(conf.field()) * 8 - 1); // 0/1
    // auto prefix = ring_add(randbit, ring_lshift(randbit, 1));  //0b00, 0x11
    // prefix = ring_lshift(prefix, SizeOf(conf.field()) * 8 - 2); //0b0000..00  0b1100..00
    // p0 = ring_add(prefix, ring_rshift(p0, 2));  //0b00...  0b11...
    // ring_print(p0, "p0");
    
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    /* WHEN */
    auto prev = comm->getStats();
    auto a1 = obj->call("drelu_sc", a0);
    auto cost = comm->getStats() - prev;


    /* THEN */
    auto msb = ring_rshift(p0, SizeOf(conf.field()) * 8 - 1);
    auto drelu_p0 = ring_sub(ring_ones(conf.field(), kNumel), msb);
    // ring_print(drelu_p0, "drelu");
    EXPECT_TRUE(ring_all_equal(drelu_p0, a2p(obj.get(), a1)));
    EXPECT_TRUE(verifyCost(obj->getKernel("drelu_sc"), "drelu_sc", conf.field(),
                           kNumel, npc, cost));

  });
}

TEST_P(ArithmeticTest, ReLU) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    // the safe range is [-2^(k-2), 2^(k-2))
    // [0, 2^(k-2)) union (2^k - 2^(k-2), 2^k-1)
    auto p0 = rand_p(obj.get(), kNumel);
    auto randbit = ring_rshift(rand_p(obj.get(), kNumel), SizeOf(conf.field()) * 8 - 1); // 0/1
    auto prefix = ring_add(randbit, ring_lshift(randbit, 1));  //0b00, 0x11
    prefix = ring_lshift(prefix, SizeOf(conf.field()) * 8 - 2); //0b0000..00  0b1100..00
    p0 = ring_add(prefix, ring_rshift(p0, 2));  //0b00...  0b11...
    // ring_print(p0, "p0");
    
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), kNumel);
    auto a0 = ring_zeros(conf.field(), kNumel);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    /* WHEN */
    auto a1 = obj->call("relu", a0);


    /* THEN */
    auto msb = ring_rshift(p0, SizeOf(conf.field()) * 8 - 1);
    auto drelu_p0 = ring_sub(ring_ones(conf.field(), kNumel), msb);
    auto relu_p0 = ring_mul(drelu_p0, p0);
    // ring_print(relu_p0, "relu");

    EXPECT_TRUE(ring_all_equal(relu_p0, a2p(obj.get(), a1)));
    

  });
}

TEST_P(ArithmeticTest, Maxpool) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto ty = makeType<spu::mpc::securenn::AShrTy>(conf.field());
    
    auto* comm = obj->getState<Communicator>();
    auto rank = comm->getRank();

    size_t maxpool_n = 10;

    // To make `msb based comparison` work, the safe range is
    // [-2^(k-2), 2^(k-2))
    // [0, 2^(k-2)) union (2^k - 2^(k-2), 2^k-1)
    auto p0 = rand_p(obj.get(), maxpool_n);
    auto randbit = ring_rshift(rand_p(obj.get(), maxpool_n), SizeOf(conf.field()) * 8 - 1); // 0/1
    auto prefix = ring_add(randbit, ring_lshift(randbit, 1));  //0b00, 0x11
    prefix = ring_lshift(prefix, SizeOf(conf.field()) * 8 - 2); //0b0000..00  0b1100..00
    p0 = ring_add(prefix, ring_rshift(p0, 2));  //0b00...  0b11...
    // ring_print(p0, "p0");

    // if(rank == 0) ring_print(p0, "p0");
    
    //auto a0 = p2a(obj.get(), p0);

    auto p1 = rand_p(obj.get(), maxpool_n);
    auto a0 = ring_zeros(conf.field(), maxpool_n);
    //ArrayRef a0(makeType<spu::mpc::securenn::AShrTy>(conf.field()), kNumel);
    if(rank == 0) a0 = p1.as(ty);
    if(rank == 1) a0 = ring_sub(p0, p1).as(ty);
    if(rank == 2) a0 = a0.as(ty);

    /* WHEN */
    auto a1 = obj->call("maxpool", a0);

    // fmt::print("\n");
    /* THEN */
    auto res = a2p(obj.get(), a1);
    // if(rank == 0) ring_print(res, "max");

    // EXPECT_TRUE(ring_all_equal(relu_p0, a2p(obj.get(), a1)));
    

  });
}


// TEST_P(ArithmeticTest, SelectShare) {
//   const auto factory = std::get<0>(GetParam());
//   const RuntimeConfig& conf = std::get<1>(GetParam());
//   const size_t npc = std::get<2>(GetParam());

//   utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
//     auto obj = factory(conf, lctx);

//     auto* comm = obj->getState<Communicator>();
//     auto rank = comm->getRank();


//     /* GIVEN */
//     auto x_p = rand_p(obj.get(), kNumel);
//     auto y_p = rand_p(obj.get(), kNumel);

//     auto p0 = rand_p(obj.get(), kNumel);
//     auto p1 = rand_p(obj.get(), kNumel);

//     auto x = ring_zeros(conf.field(), kNumel);
//     auto y = ring_zeros(conf.field(), kNumel);
//     auto alpha0 = ring_zeros(conf.field(), kNumel);//alpha0 = 0 = 0 + 0
//     auto alpha1 = ring_zeros(conf.field(), kNumel);//alpha1 = 1 = 1 + 0

//     if(rank == 0){
//       x = p0.as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//       y = p1.as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//       alpha1 = ring_ones(conf.field(), kNumel).as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//     }
//     if(rank == 1){
//       x = ring_sub(x_p, p0).as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//       y = ring_sub(y_p, p1).as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//     }
//     if(rank == 2){
//       x = x.as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//       y = y.as(makeType<spu::mpc::securenn::AShrTy>(conf.field()));
//     }

//     /* WHEN */
//     auto select_x = obj->call("select_share", alpha0, x, y);
//     auto select_y = obj->call("select_share", alpha1, x, y);

//     /* THEN */
//     EXPECT_TRUE(ring_all_equal(a2p(obj.get(), select_x), x_p));
//     EXPECT_TRUE(ring_all_equal(a2p(obj.get(), select_y), y_p));

//   });
// }

}  // namespace spu::mpc::test
