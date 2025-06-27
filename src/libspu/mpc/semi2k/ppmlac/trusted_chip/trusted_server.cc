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

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_server.h"

#include "brpc/server.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/crypto/pke/sm2_enc.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/semi2k/ppmlac/trusted_chip/stream_reader.h"
#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_chip.h"

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_service.pb.h"

namespace spu::mpc::semi2k::ppmlac {

class TrustedServiceImpl final : public TrustedService {
 public:
  TrustedServiceImpl(const std::string& asym_crypto_schema,
                     yacl::ByteContainerView public_key,
                     yacl::ByteContainerView private_key)
      : chip_(asym_crypto_schema, public_key, private_key) {}

  void GetPubKey(::google::protobuf::RpcController* controller,
                 const GetPubKeyRequest* request, GetPubKeyResponse* response,
                 ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_asym_crypto_schema(chip_.GetAsymCryptoSchema());
    response->set_pub_key(chip_.GetPubKey().data(), chip_.GetPubKey().size());
  }

  void ExRandNum(::google::protobuf::RpcController* controller,
                 const ExRandNumRequest* request, ExRandNumResponse* response,
                 ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    // TODO: add lock
    chip_.SetupPRNG(request->rank(), request->enc_rand_num());

    yacl::Buffer pk = chip_.DecryptPubKey(request->enc_pub_key());
    response->set_enc_rand_num(chip_.EncryptRandNum(pk));
  }

  void Mul(::google::protobuf::RpcController* controller,
           const MulRequest* request, Response* response,
           ::google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    // StreamReader reader(
    //     2, SizeOf(request->field_type()) * request->num_elements());

    brpc::StreamOptions stream_options;
    stream_options.max_buf_size = 0;
    // stream_options.handler = &reader;
    auto* cntl = static_cast<brpc::Controller*>(controller);
    brpc::StreamId stream_id = brpc::INVALID_STREAM_ID;
    if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
      SPDLOG_ERROR("Failed to accept stream");
      response->set_code(ErrorCode::StreamAcceptError);
      return;
    }
  }

 private:
  TrustedChip chip_;
};

std::unique_ptr<brpc::Server> RunServer(const TrustedServerOptions& options) {
  // brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  // brpc::FLAGS_socket_max_unwritten_bytes =
  //     std::numeric_limits<int64_t>::max() / 2;

  auto server = std::make_unique<brpc::Server>();
  auto svc = std::make_unique<TrustedServiceImpl>(
      options.asym_crypto_schema, options.public_key, options.private_key);

  if (server->AddService(svc.release(), brpc::SERVER_OWNS_SERVICE) != 0) {
    SPDLOG_ERROR("Fail to add service");
    return nullptr;
  }

  // workaround fix
  brpc::ServerOptions brpc_options = server->options();

  if (options.brpc_ssl_options) {
    *brpc_options.mutable_ssl_options() = options.brpc_ssl_options.value();
  }

  brpc_options.has_builtin_services = false;
  if (server->Start(options.port, &brpc_options) != 0) {
    SPDLOG_ERROR("Fail to start Server");
    return nullptr;
  }

  return server;
}

int RunUntilAskedToQuit(const TrustedServerOptions& options) {
  auto server = RunServer(options);

  SPU_ENFORCE(server);

  server->RunUntilAskedToQuit();

  return 0;
}

}  // namespace spu::mpc::semi2k::ppmlac
