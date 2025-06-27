// Copyright 2025 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "trusted_chip/trusted_server.h"
#include "yacl/crypto/key_utils.h"

#include "libspu/mpc/semi2k/ppmlac/ppmlac_factory.h"
#include "libspu/mpc/semi2k/ppmlac/ppmlac_receiver.h"
#include "libspu/mpc/semi2k/ppmlac/ppmlac_sender.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/spu.h"

namespace spu::mpc::semi2k::ppmlac {

class PPMLACTest
    : public testing::TestWithParam<std::tuple<size_t, FieldType, size_t>> {
 protected:
  static void SetUpTestSuite() {
    {
      server_options_.port = 0;
      server_options_.asym_crypto_schema = "sm2";
      auto [pk, sk] = yacl::crypto::GenSm2KeyPairToPemBuf();
      server_options_.public_key = pk;
      server_options_.private_key = sk;
    }
    {
      auto [pk, sk] = yacl::crypto::GenSm2KeyPairToPemBuf();
      public_key_ = pk;
      private_key_ = sk;
    }
  }

  void SetUp() override { server_ = RunServer(server_options_); }

  void TearDown() override {
    server_->Stop(0);
    server_.reset();
  }

  std::unique_ptr<PPMLAC> InitalizePPMLAC(
      const std::shared_ptr<yacl::link::Context>& lctx, size_t recv_rank) {
    if (lctx->Rank() == recv_rank) {
      ReceiverPPMLAC::Options options;
      options.recv_rank = recv_rank;
      options.trusted_server_host =
          "127.0.0.1:" + std::to_string(server_->listen_address().port);
      auto ppmlac = PPMLACFactory::CreatePPMLAC(options, lctx);
      ppmlac->Initialize();
      return ppmlac;
    } else {
      SenderPPMLAC::Options options;
      options.recv_rank = recv_rank;
      options.asym_crypto_schema = "sm2";
      options.pub_key = public_key_;
      options.priv_key = private_key_;
      auto ppmlac = PPMLACFactory::CreatePPMLAC(options, lctx);
      ppmlac->Initialize();
      return ppmlac;
    }
  }

  std::unique_ptr<brpc::Server> server_;
  static TrustedServerOptions server_options_;

  static yacl::Buffer public_key_;
  static yacl::Buffer private_key_;
};

yacl::Buffer PPMLACTest::public_key_;
yacl::Buffer PPMLACTest::private_key_;
TrustedServerOptions PPMLACTest::server_options_;

INSTANTIATE_TEST_SUITE_P(
    PPMLAC, PPMLACTest,
    testing::Combine(testing::Values(3), testing::Values(FieldType::FM64),
                     testing::Values(0)),
    [](const testing::TestParamInfo<PPMLACTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         int(std::get<1>(p.param)), std::get<2>(p.param));
    });

TEST_P(PPMLACTest, Initialize) {
  size_t world_size = std::get<0>(GetParam());
  // FieldType field = std::get<1>(GetParam());
  size_t recv_rank = std::get<2>(GetParam());

  auto fn = [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ppmlac = InitalizePPMLAC(lctx, recv_rank);
  };

  utils::simulate(world_size, fn);
}

}  // namespace spu::mpc::semi2k::ppmlac
