// Copyright 2021 Ant Group Co., Ltd.
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

#include <memory>

#include "libspu/core/object.h"
#include "libspu/mpc/semi2k/beaver/beaver_cache.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_tfp.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_ttp.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"
#include "libspu/mpc/semi2k/ppmlac/ppmlac_receiver.h"
#include "libspu/mpc/semi2k/ppmlac/ppmlac_sender.h"

namespace spu::mpc {

// TODO(jint) split this into individual states.
class Semi2kState : public State {
  std::unique_ptr<semi2k::Beaver> beaver_;
  std::shared_ptr<semi2k::BeaverCache> beaver_cache_;
  std::unique_ptr<semi2k::ppmlac::PPMLAC> ppmlac_;

 private:
  Semi2kState() = default;

 public:
  static constexpr const char* kBindName() { return "Semi2kState"; }

  explicit Semi2kState(const RuntimeConfig& conf,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
    if (conf.ppmlac_config.has_value()) {
      InitPPMLAC(conf, lctx);
    } else {
      InitBeaver(conf, lctx);
    }
  }

  semi2k::Beaver* beaver() { return beaver_.get(); }
  semi2k::BeaverCache* beaver_cache() { return beaver_cache_.get(); }
  semi2k::ppmlac::PPMLAC* ppmlac() { return ppmlac_.get(); }

  std::unique_ptr<State> fork() override {
    auto ret = std::unique_ptr<Semi2kState>(new Semi2kState);
    ret->beaver_ = beaver_->Spawn();
    ret->beaver_cache_ = beaver_cache_;
    return ret;
  }

 private:
  void InitBeaver(const RuntimeConfig& conf,
                  const std::shared_ptr<yacl::link::Context>& lctx) {
    if (conf.beaver_type == RuntimeConfig::TrustedFirstParty) {
      beaver_ = std::make_unique<semi2k::BeaverTfpUnsafe>(lctx);
    } else if (conf.beaver_type == RuntimeConfig::TrustedThirdParty) {
      semi2k::BeaverTtp::Options ops;
      SPU_ENFORCE(conf.has_ttp_beaver_config());
      ops.server_host = conf.ttp_beaver_config->server_host;
      ops.adjust_rank = conf.ttp_beaver_config->adjust_rank;
      ops.asym_crypto_schema = conf.ttp_beaver_config->asym_crypto_schema;
      {
        const auto& key = conf.ttp_beaver_config->server_public_key;
        ops.server_public_key = yacl::Buffer(key.data(), key.size());
      }
      if (!conf.ttp_beaver_config->transport_protocol.empty()) {
        ops.brpc_channel_protocol = conf.ttp_beaver_config->transport_protocol;
      }
      if (conf.ttp_beaver_config->has_ssl_config()) {
        auto& ssl_config = conf.ttp_beaver_config->ssl_config;
        brpc::ChannelSSLOptions ssl_options;
        ssl_options.verify.ca_file_path = ssl_config->ca_file_path;
        ssl_options.verify.verify_depth = ssl_config->verify_depth;
        ssl_options.client_cert.certificate = ssl_config->certificate;
        ssl_options.client_cert.private_key = ssl_config->private_key;
        ops.brpc_ssl_options = std::move(ssl_options);
      }
      beaver_ = std::make_unique<semi2k::BeaverTtp>(lctx, std::move(ops));
    } else {
      SPU_THROW("unsupported beaver type {}", conf.beaver_type);
    }
    beaver_cache_ = std::make_unique<semi2k::BeaverCache>();
  }

  void InitPPMLAC(const RuntimeConfig& conf,
                  const std::shared_ptr<yacl::link::Context>& lctx) {
    if (lctx->Rank() == conf.ppmlac_config->receiver_rank) {
      semi2k::ppmlac::ReceiverPPMLAC::Options options;
      options.recv_rank = conf.ppmlac_config->receiver_rank;
      options.trusted_server_host = conf.ppmlac_config->server_host;
      if (conf.ppmlac_config->ssl_config.has_value()) {
        const auto& ssl_config = conf.ppmlac_config->ssl_config.value();
        brpc::ChannelSSLOptions ssl_options;
        ssl_options.verify.ca_file_path = ssl_config.ca_file_path;
        ssl_options.verify.verify_depth = ssl_config.verify_depth;
        ssl_options.client_cert.certificate = ssl_config.certificate;
        ssl_options.client_cert.private_key = ssl_config.private_key;
        options.brpc_ssl_options = std::move(ssl_options);
      }
      ppmlac_ = std::make_unique<semi2k::ppmlac::ReceiverPPMLAC>(lctx, options);
    } else {
      semi2k::ppmlac::SenderPPMLAC::Options options;
      options.recv_rank = conf.ppmlac_config->receiver_rank;
      options.asym_crypto_schema = conf.ppmlac_config->asym_crypto_schema;
      options.pub_key = yacl::Buffer(conf.ppmlac_config->public_key);
      options.priv_key = yacl::Buffer(conf.ppmlac_config->private_key);
      ppmlac_ = std::make_unique<semi2k::ppmlac::SenderPPMLAC>(lctx, options);
    }
    ppmlac_->Initialize();
  }
};

}  // namespace spu::mpc
