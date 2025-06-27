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

#pragma once

#include "brpc/channel.h"

#include "libspu/mpc/semi2k/ppmlac/ppmlac.h"

namespace spu::mpc::semi2k::ppmlac {

class ReceiverPPMLAC : public PPMLAC {
 public:
  struct Options : PPMLAC::Options {
    std::string trusted_server_host;
    std::string brpc_channel_protocol = "baidu_std";
    int32_t brpc_connect_timeout_ms = 10 * 1000;
    int32_t brpc_timeout_ms = 300 * 1000;
    int32_t brpc_max_retry = 5;
    std::optional<brpc::ChannelSSLOptions> brpc_ssl_options;
  };

  ReceiverPPMLAC(const std::shared_ptr<yacl::link::Context>& lctx,
                 const Options& options)
      : PPMLAC(lctx) {
    InitChannel(options);
  }

  ~ReceiverPPMLAC() override = default;

  void Initialize() override;

  NdArrayRef Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                 const NdArrayRef& y) override;

 private:
  void InitChannel(const Options& options);

  // Query public key from the trusted server
  std::tuple<std::string, yacl::Buffer> QueryPubKey() const;

  // Exchange random number with the trusted server
  yacl::Buffer ExRandNum(size_t rank, yacl::ByteContainerView enc_pk,
                         yacl::ByteContainerView enc_rn) const;

  mutable brpc::Channel channel_;  // TODO: mutable?
};

}  // namespace spu::mpc::semi2k::ppmlac
