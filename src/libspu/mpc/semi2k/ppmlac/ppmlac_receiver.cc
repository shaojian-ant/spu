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

#include "libspu/mpc/semi2k/ppmlac/ppmlac_receiver.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_service.pb.h"

namespace spu::mpc::semi2k::ppmlac {

void ReceiverPPMLAC::InitChannel(const Options& options) {
  // brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  // brpc::FLAGS_socket_max_unwritten_bytes =
  //     std::numeric_limits<int64_t>::max() / 2;
  brpc::ChannelOptions brc_options;
  SPU_ENFORCE(options.brpc_channel_protocol == "http" ||
                  options.brpc_channel_protocol == "h2",
              "Only support http 1.1 or http 2");
  brc_options.protocol = options.brpc_channel_protocol;
  brc_options.timeout_ms = options.brpc_timeout_ms;
  brc_options.max_retry = options.brpc_max_retry;

  if (options.brpc_ssl_options) {
    *brc_options.mutable_ssl_options() = options.brpc_ssl_options.value();
  }

  if (channel_.Init(options.trusted_server_host.c_str(), &brc_options) != 0) {
    SPU_THROW("Fail to initialize channel for PPMLAC receiver, server_host {}",
              options.trusted_server_host);
  }
}

std::tuple<std::string, yacl::Buffer> ReceiverPPMLAC::QueryPubKey() const {
  GetPubKeyRequest request;
  GetPubKeyResponse response;
  brpc::Controller cntl;
  TrustedService_Stub stub(&channel_);
  stub.GetPubKey(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(), "GetPubKey RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());
  return {response.asym_crypto_schema(), yacl::Buffer(response.pub_key())};
}

// Exchange random number with the trusted server
yacl::Buffer ReceiverPPMLAC::ExRandNum(size_t rank,
                                       yacl::ByteContainerView enc_pk,
                                       yacl::ByteContainerView enc_rn) const {
  ExRandNumRequest request;
  request.set_rank(rank);
  request.set_enc_pub_key(enc_pk.data(), enc_pk.size());
  request.set_enc_rand_num(enc_rn.data(), enc_rn.size());

  brpc::Controller cntl;
  ExRandNumResponse response;
  TrustedService_Stub stub(&channel_);
  stub.ExRandNum(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(), "ExRandNum RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());

  return {response.enc_rand_num().data(), response.enc_rand_num().size()};
}

yacl::Buffer MulRpcCall(brpc::Channel& channel, const NdArrayRef& u,
                        const NdArrayRef& v) {
  const auto field = u.eltype().as<Ring2k>()->field();
  TrustedService_Stub stub(&channel);
  brpc::Controller cntl;
  brpc::StreamId stream;
  SPU_ENFORCE_EQ(brpc::StreamCreate(&stream, cntl, nullptr), 0,
                 "Fail to create stream");

  MulRequest request;
  Response response;
  request.set_field_type(static_cast<ppmlac::FieldType>(field));
  request.set_num_elements(u.numel());
  stub.Mul(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(),
              "Mul RpcCall failed, cntl.ErrorCode={} cntl.ErrorText={}",
              cntl.ErrorCode(), cntl.ErrorText());
  SPU_ENFORCE(response.code() == 0,
                 "Mul RpcCall failed, response.code={} response.message={}",
                 static_cast<int>(response.code()), response.message());
  return yacl::Buffer{};

  // SPU_ENFORCE_EQ(brpc::StreamWrite(stream_id, message), 0);
  // SPDLOG_DEBUG("write buf size {} to stream id {}", message.length(),
  //              stream_id);
}

void ReceiverPPMLAC::Initialize() {
  auto [_, pub_key] = QueryPubKey();

  // TODO: multi-thread?
  for (size_t i = 0; i < lctx_->WorldSize(); ++i) {
    if (i == lctx_->Rank()) {
      continue;
    }

    // Send public key to rank i
    lctx_->SendAsync(i, pub_key, "PPMLAC:pub_key");
    SPDLOG_INFO("Send public key to rank {}", i);

    // Recv encrypted public key from rank i
    auto enc_pk = lctx_->Recv(i, "PPMLAC:pub_key");
    SPDLOG_INFO("Recv encrypted public key from rank {}", i);

    // Recv encrypted rand num from rank i
    auto enc_rn = lctx_->Recv(i, "PPMLAC:rand_num");
    SPDLOG_INFO("Recv encrypted random number from rank {}", i);

    // Exchange random number with rank i
    auto self_enc_rn = ExRandNum(i, enc_pk, enc_rn);

    // Send encrypted random number to rank i
    lctx_->SendAsync(i, self_enc_rn, "PPMLAC:rand_num");
    SPDLOG_INFO("Send encrypted random number to rank {}", i);
  }
}

NdArrayRef ReceiverPPMLAC::Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  auto res = vmap({x, y}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, lctx_->Rank(), "recv(x-a,y-b)");
  });
  auto u = std::move(res[0]);
  auto v = std::move(res[1]);

  return u;
}

}  // namespace spu::mpc::semi2k::ppmlac
