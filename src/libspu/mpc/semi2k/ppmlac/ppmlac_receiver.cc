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

#include "trusted_chip/stream_reader.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/ppmlac/bit_set.h"
#include "libspu/mpc/semi2k/ppmlac/utils.h"
#include "libspu/mpc/semi2k/type.h"

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_service.pb.h"

namespace brpc {

DECLARE_uint64(max_body_size);
DECLARE_int64(socket_max_unwritten_bytes);

}  // namespace brpc

namespace spu::mpc::semi2k::ppmlac {

namespace {

// Query public key from the trusted server
std::tuple<std::string, yacl::Buffer> QueryPubKey(brpc::Channel& channel) {
  GetPubKeyRequest request;
  GetPubKeyResponse response;
  brpc::Controller cntl;
  TrustedService_Stub stub(&channel);
  stub.GetPubKey(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(), "GetPubKey RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());
  return {response.asym_crypto_schema(), yacl::Buffer(response.pub_key())};
}

// Exchange random number with the trusted server
yacl::Buffer ExRandNum(brpc::Channel& channel, size_t rank,
                       yacl::ByteContainerView enc_pk,
                       yacl::ByteContainerView enc_rn) {
  ExRandNumRequest request;
  request.set_rank(rank);
  request.set_enc_pub_key(enc_pk.data(), enc_pk.size());
  request.set_enc_rand_num(enc_rn.data(), enc_rn.size());

  brpc::Controller cntl;
  ExRandNumResponse response;
  TrustedService_Stub stub(&channel);
  stub.ExRandNum(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(), "ExRandNum RpcCall failed, code={} error={}",
              cntl.ErrorCode(), cntl.ErrorText());

  return {response.enc_rand_num().data(), response.enc_rand_num().size()};
}

template <typename Request, typename... Args>
Request BuildRequest(FieldType field, const Shape& shape, Args&&... args) {
  Request req;
  req.set_field(static_cast<Field>(field));
  if constexpr (std::is_same_v<Request, DotRequest>) {
    req.set_m(shape.at(0));
    req.set_n(shape.at(1));
    req.set_k(shape.at(2));
  } else if constexpr (std::is_same_v<Request, TruncRequest>) {
    req.set_num_elements(shape.numel());
    req.set_bits(args...);
  } else if constexpr (std::is_same_v<Request, B2ARequest>) {
    req.set_num_elements(shape.numel());
    req.set_bshare_nbits(args...);
  } else if constexpr (std::is_same_v<Request, PermRequest>) {
    req.set_num_elements(shape.numel());
    req.set_perm_rank(args...);
  } else {
    req.set_num_elements(shape.numel());
  }
  return req;
}

template <typename Request>
Shape GetReturnShape(const Shape& shape) {
  if constexpr (std::is_same_v<Request, DotRequest>) {
    const int64_t m = shape.at(0);
    const int64_t n = shape.at(1);
    return {m, n};
  } else {
    return shape;
  }
}

template <typename Request>
size_t GetReturnBufferLength(FieldType field, const Shape& shape) {
  if constexpr (std::is_same_v<Request, DotRequest>) {
    const int64_t m = shape.at(0);
    const int64_t n = shape.at(1);
    return SizeOf(field) * m * n;
  } else if constexpr (std::is_same_v<Request, EqzRequest>) {
    return CeilDiv(shape.numel(), 8);
  } else {
    return {SizeOf(field) * shape.numel()};
  }
}

template <typename Request>
NdArrayRef GenerateReturnArray(const butil::IOBuf& buf, FieldType field,
                               const Shape& shape) {
  if constexpr (std::is_same_v<Request, EqzRequest>) {
    NdArrayRef out(makeType<BShrTy>(field), shape);
    BitSet bs(shape.numel());
    buf.copy_to(bs.Data());
    DISPATCH_ALL_FIELDS(field, [&]() {
      using el_t = ring2k_t;
      // 1 bit info in lsb
      NdArrayView<el_t> _out(out);
      pforeach(0, shape.numel(), [&](int64_t idx) { _out[idx] = bs.At(idx); });
    });
    return out;
  } else {
    NdArrayRef out(makeType<AShrTy>(field), shape);
    buf.copy_to(out.data<void>());
    return out;
  }
}

template <typename Request>
using RpcFunc = void (TrustedService_Stub::*)(
    ::google::protobuf::RpcController*, const Request*, Response*,
    ::google::protobuf::Closure*);

template <typename Request, typename... Args>
NdArrayRef RpcCall(RpcFunc<Request> pf, brpc::Channel& channel, FieldType field,
                   const Shape& shape, absl::Span<NdArrayRef> inputs,
                   Args&&... args) {
  brpc::Controller cntl;
  brpc::StreamId stream_id;
  StreamReader reader({GetReturnBufferLength<Request>(field, shape)});
  brpc::StreamOptions stream_options;
  stream_options.max_buf_size = 0;  // there's no limit of buf size
  stream_options.handler = &reader;
  SPU_ENFORCE_EQ(brpc::StreamCreate(&stream_id, cntl, &stream_options), 0,
                 "Fail to create stream");
  SPDLOG_DEBUG("Stream {} created", stream_id);

  Response response;
  TrustedService_Stub stub(&channel);
  auto request =
      BuildRequest<Request>(field, shape, std::forward<Args>(args)...);
  (stub.*pf)(&cntl, &request, &response, nullptr);
  SPU_ENFORCE(!cntl.Failed(),
              "Mul RpcCall failed, cntl.ErrorCode={} cntl.ErrorText={}",
              cntl.ErrorCode(), cntl.ErrorText());
  SPU_ENFORCE(response.code() == 0,
              "Mul RpcCall failed, response.code={} response.message={}",
              static_cast<int>(response.code()), response.message());

  // Send u and v to trusted chip
  for (auto input : inputs) {
    SPU_ENFORCE(input.isCompact());
    SendMessage(stream_id, input.data(), input.numel() * input.elsize());
  }

  // Receive z from trusted chip
  reader.WaitFinished();
  SPU_ENFORCE_EQ(brpc::StreamClose(stream_id), 0);
  const auto& bufs = reader.GetBufVecRef();
  auto z = GenerateReturnArray<Request>(bufs.at(0), field,
                                        GetReturnShape<Request>(shape));
  reader.WaitClosed();  // TODO: move it
  return z;
}

}  // namespace

void ReceiverPPMLAC::InitChannel(const Options& options) {
  brpc::FLAGS_max_body_size = std::numeric_limits<uint64_t>::max();
  brpc::FLAGS_socket_max_unwritten_bytes =
      std::numeric_limits<int64_t>::max() / 2;
  brpc::ChannelOptions brpc_options;
  SPU_ENFORCE(options.brpc_channel_protocol == "baidu_std",
              "Only the baidu_std protocol is supported for brpc streaming");
  brpc_options.protocol = options.brpc_channel_protocol;
  brpc_options.timeout_ms = options.brpc_timeout_ms;
  brpc_options.max_retry = options.brpc_max_retry;

  if (options.brpc_ssl_options) {
    *brpc_options.mutable_ssl_options() = options.brpc_ssl_options.value();
  }

  if (channel_.Init(options.trusted_server_host.c_str(), &brpc_options) != 0) {
    SPU_THROW("Fail to initialize channel for PPMLAC receiver, server_host {}",
              options.trusted_server_host);
  }
}

void ReceiverPPMLAC::Initialize() {
  auto [_, pub_key] = QueryPubKey(channel_);

  // TODO: multi-thread?
  for (size_t i = 0; i < lctx_->WorldSize(); ++i) {
    if (i == lctx_->Rank()) {
      continue;
    }

    // Send public key to rank i
    lctx_->SendAsync(i, pub_key, "PPMLAC:pub_key");
    SPDLOG_DEBUG("Send public key to rank {}", i);

    // Recv encrypted public key from rank i
    auto enc_pk = lctx_->Recv(i, "PPMLAC:pub_key");
    SPDLOG_DEBUG("Recv encrypted public key from rank {}", i);

    // Recv encrypted rand num from rank i
    auto enc_rn = lctx_->Recv(i, "PPMLAC:rand_num");
    SPDLOG_DEBUG("Recv encrypted random number from rank {}", i);

    // Exchange random number with rank i
    auto self_enc_rn = ExRandNum(channel_, i, enc_pk, enc_rn);

    // Send encrypted random number to rank i
    lctx_->SendAsync(i, self_enc_rn, "PPMLAC:rand_num");
    SPDLOG_DEBUG("Send encrypted random number to rank {}", i);
  }

  SPDLOG_DEBUG("PPMLAC finished initialization with rank {}", lctx_->Rank());
}

NdArrayRef ReceiverPPMLAC::Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto inputs = vmap({x, y}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, comm->getRank(), "recv(x-a,y-b)");
  });

  return RpcCall(&TrustedService_Stub::Mul, channel_, field, x.shape(),
                 absl::MakeSpan(inputs));
}

NdArrayRef ReceiverPPMLAC::Square(KernelEvalContext* ctx, const NdArrayRef& x) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  NdArrayRef u = comm->reduce(ReduceOp::ADD, x, comm->getRank(), "recv(x-a)");

  return RpcCall(&TrustedService_Stub::Square, channel_, field, x.shape(),
                 absl::MakeSpan(&u, 1));
}

NdArrayRef ReceiverPPMLAC::MatMul(KernelEvalContext* ctx, const NdArrayRef& x,
                                  const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto inputs = vmap({x, y}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, comm->getRank(), "recv(x-a,y-b)");
  });

  const int64_t m = x.shape().at(0);
  const int64_t n = y.shape().at(1);
  const int64_t k = x.shape().at(1);

  return RpcCall(&TrustedService_Stub::Dot, channel_, field, {m, n, k},
                 absl::MakeSpan(inputs));
}

NdArrayRef ReceiverPPMLAC::And(KernelEvalContext* ctx, const NdArrayRef& lhs,
                               const NdArrayRef& rhs) {
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  SPU_ENFORCE(lhs.eltype().as<Ring2k>()->field() ==
              rhs.eltype().as<Ring2k>()->field());

  auto* comm = ctx->getState<Communicator>();
  const auto field = lhs.eltype().as<Ring2k>()->field();

  const size_t out_nbits = std::min(lhs.eltype().as<BShrTy>()->nbits(),
                                    rhs.eltype().as<BShrTy>()->nbits());

  const PtType backtype = getBacktype(out_nbits);
  const int64_t numel = lhs.numel();

  int64_t num128 = CeilDiv(numel * SizeOf(backtype), SizeOf(FieldType::FM128));

  NdArrayRef out(makeType<BShrTy>(field, out_nbits), lhs.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _lhs(lhs);
    NdArrayView<T> _rhs(rhs);

    DISPATCH_UINT_PT_TYPES(backtype, [&]() {
      using V = ScalarT;

      // first half mask x^a, second half mask y^b.
      std::vector<V> mask(numel * 2, 0);

      mask =
          comm->reduce<V, std::bit_xor>(mask, comm->getRank(), "open(x^a,y^b)");

      std::vector<NdArrayRef> inputs = {
          NdArrayRef(makeType<RingTy>(FieldType::FM128), {num128}),
          NdArrayRef(makeType<RingTy>(FieldType::FM128), {num128})};

      // first half mask x^a, second half mask y^b.
      pforeach(0, numel, [&](int64_t idx) {
        inputs[0].data<V>()[idx] = _lhs[idx] ^ mask[idx];
        inputs[1].data<V>()[idx] = _rhs[idx] ^ mask[numel + idx];
      });

      auto ret = RpcCall(&TrustedService_Stub::And, channel_, FieldType::FM128,
                         {num128}, absl::MakeSpan(inputs));

      NdArrayView<T> _out(out);
      pforeach(0, numel, [&](int64_t idx) { _out[idx] = ret.data<V>()[idx]; });
    });
  });

  return out;
}

NdArrayRef ReceiverPPMLAC::Trunc(KernelEvalContext* ctx, const NdArrayRef& x,
                                 size_t bits) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  NdArrayRef u = comm->reduce(ReduceOp::ADD, x, comm->getRank(), "recv(x-a)");

  return RpcCall(&TrustedService_Stub::Trunc, channel_, field, x.shape(),
                 absl::MakeSpan(&u, 1), bits);
}

NdArrayRef ReceiverPPMLAC::B2A(KernelEvalContext* ctx, const NdArrayRef& x) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  const size_t out_nbits = x.eltype().as<BShrTy>()->nbits();
  const PtType backtype = getBacktype(out_nbits);
  const int64_t numel = x.numel();

  int64_t num128 = CeilDiv(numel * SizeOf(backtype), SizeOf(FieldType::FM128));
  NdArrayRef u(makeType<RingTy>(FieldType::FM128), {num128});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _x(x);

    DISPATCH_UINT_PT_TYPES(backtype, [&]() {
      using V = ScalarT;
      NdArrayView<V> _u(u);

      std::vector<V> mask(numel, 0);
      mask = comm->reduce<V, std::bit_xor>(mask, comm->getRank(), "open(x^a)");

      pforeach(0, numel, [&](int64_t idx) { _u[idx] = _x[idx] ^ mask[idx]; });
    });
  });

  return RpcCall(&TrustedService_Stub::B2A, channel_, field, x.shape(),
                 absl::MakeSpan(&u, 1), out_nbits);
}

NdArrayRef ReceiverPPMLAC::Eqz(KernelEvalContext* ctx, const NdArrayRef& z) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = z.eltype().as<Ring2k>()->field();
  NdArrayRef u = comm->reduce(ReduceOp::ADD, z, comm->getRank(), "recv(z-a)");

  return RpcCall(&TrustedService_Stub::Eqz, channel_, field, z.shape(),
                 absl::MakeSpan(&u, 1));
}

NdArrayRef ReceiverPPMLAC::Perm(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& perm, size_t perm_rank) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  std::vector<NdArrayRef> inputs(2);
  inputs[0] = comm->reduce(ReduceOp::ADD, x, comm->getRank(), "recv(x-a)");

  if (perm_rank == lctx_->Rank()) {
    inputs[1] = perm;
  } else {
    inputs[1] = comm->recv(perm_rank, makeType<PShrTy>(), "recv(perm_mask)");
  }

  return RpcCall(&TrustedService_Stub::Perm, channel_, field, x.shape(),
                 absl::MakeSpan(inputs), perm_rank);
}

}  // namespace spu::mpc::semi2k::ppmlac
