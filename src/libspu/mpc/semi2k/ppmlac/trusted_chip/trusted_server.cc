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

#include "absl/functional/bind_front.h"
#include "brpc/server.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/crypto/pke/sm2_enc.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/semi2k/ppmlac/trusted_chip/stream_reader.h"
#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_chip.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_service.pb.h"

namespace spu::mpc::semi2k::ppmlac {

namespace {

template <typename Request>
std::vector<size_t> GetBufferLength(const Request& req) {
  size_t field_size = SizeOf(static_cast<spu::FieldType>(req.field_type()));
  if constexpr (std::is_same_v<Request, MulRequest> ||
                std::is_same_v<Request, AndRequest>) {
    return std::vector<size_t>(2, req.num_elements() * field_size);
  } else if constexpr (std::is_same_v<Request, SquareRequest> ||
                       std::is_same_v<Request, TruncRequest> ||
                       std::is_same_v<Request, EqzRequest>) {
    return {req.num_elements() * field_size};
  } else if constexpr (std::is_same_v<Request, DotRequest>) {
    return {req.m() * req.k() * field_size, req.k() * req.n() * field_size};
  } else {
    SPU_THROW("Not supported request type");
  }
}

template <typename Request>
std::tuple<brpc::StreamId, std::shared_ptr<StreamReader>> CreateStreamReader(
    brpc::Controller* cntl, const Request* request) {
  auto reader = std::make_shared<StreamReader>(GetBufferLength(*request));
  brpc::StreamOptions stream_options;
  stream_options.max_buf_size = 0;
  stream_options.handler = reader.get();

  brpc::StreamId stream_id;
  if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
    SPDLOG_ERROR("Failed to accept stream");
    cntl->SetFailed("Failed to accept stream");
    return std::make_tuple(brpc::INVALID_STREAM_ID, std::move(reader));
  }
  SPDLOG_DEBUG("Stream {} created", stream_id);

  return std::make_tuple(stream_id, std::move(reader));
}

}  // namespace

class TrustedServiceImpl final : public TrustedService {
 public:
  TrustedServiceImpl(const std::string& asym_crypto_schema,
                     yacl::ByteContainerView public_key,
                     yacl::ByteContainerView private_key)
      : chip_(asym_crypto_schema, public_key, private_key) {}

  void GetPubKey(::google::protobuf::RpcController* controller,
                 const GetPubKeyRequest* request, GetPubKeyResponse* response,
                 ::google::protobuf::Closure* done) override {
    SPDLOG_DEBUG("Process GetPubKey request");
    brpc::ClosureGuard done_guard(done);
    response->set_asym_crypto_schema(chip_.GetAsymCryptoSchema());
    response->set_pub_key(chip_.GetPubKey().data(), chip_.GetPubKey().size());
  }

  void ExRandNum(::google::protobuf::RpcController* controller,
                 const ExRandNumRequest* request, ExRandNumResponse* response,
                 ::google::protobuf::Closure* done) override {
    SPDLOG_DEBUG("Process ExRandNum request");
    brpc::ClosureGuard done_guard(done);

    // TODO: add lock
    chip_.SetupPRNG(request->rank(), request->enc_rand_num());

    yacl::Buffer pk = chip_.DecryptPubKey(request->enc_pub_key());
    response->set_enc_rand_num(chip_.EncryptRandNum(pk));
  }

  void Mul(::google::protobuf::RpcController* controller,
           const MulRequest* request, Response* response,
           ::google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  void Square(::google::protobuf::RpcController* controller,
              const SquareRequest* request, Response* response,
              ::google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  void Dot(::google::protobuf::RpcController* controller,
           const DotRequest* request, Response* response,
           ::google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  void And(google::protobuf::RpcController* controller,
           const AndRequest* request, Response* response,
           google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  void Trunc(::google::protobuf::RpcController* controller,
             const TruncRequest* request, Response* response,
             ::google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  void Eqz(google::protobuf::RpcController* controller,
           const EqzRequest* request, Response* response,
           google::protobuf::Closure* done) override {
    HandleRequest(controller, request, done);
  }

  template <typename Request>
  void HandleRequest(::google::protobuf::RpcController* controller,
                     const Request* request,
                     ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    auto [stream_id, reader] = CreateStreamReader(cntl, request);

    cntl->set_after_rpc_resp_fn(absl::bind_front(
        &TrustedServiceImpl::CallAfterRpc, this, stream_id, reader));
    // TODO: why std::unique<StreamReader> make compiling failed?
  }

  void CallAfterRpc(brpc::StreamId stream_id,
                    const std::shared_ptr<StreamReader>& reader,
                    brpc::Controller* cntl,
                    const google::protobuf::Message* req,
                    const google::protobuf::Message* res) {
    reader->WaitFinished();
    const auto& bufs = reader->GetBufVecRef();

    const std::string& type_name = req->GetDescriptor()->full_name();
    if (type_name == EqzRequest::descriptor()->full_name()) {
      auto bit_set = EqzImpl(static_cast<const EqzRequest*>(req), bufs);
      SendMessage(stream_id, bit_set.Data(), bit_set.SizeInBytes());
    } else {
      NdArrayRef z;
      if (type_name == MulRequest::descriptor()->full_name()) {
        z = MulImpl(static_cast<const MulRequest*>(req), bufs);
      } else if (type_name == SquareRequest::descriptor()->full_name()) {
        z = SquareImpl(static_cast<const SquareRequest*>(req), bufs);
      } else if (type_name == DotRequest::descriptor()->full_name()) {
        z = DotImpl(static_cast<const DotRequest*>(req), bufs);
      } else if (type_name == AndRequest::descriptor()->full_name()) {
        z = AndImpl(static_cast<const AndRequest*>(req), bufs);
      } else if (type_name == TruncRequest::descriptor()->full_name()) {
        z = TruncImpl(static_cast<const TruncRequest*>(req), bufs);
      } else {
        SPU_THROW("Not supported request type");
      }

      SPU_ENFORCE(z.isCompact());
      SendMessage(stream_id, z.data(), z.numel() * z.elsize());
    }

    reader->WaitClosed();  // TODO: move it
  }

NdArrayRef MulImpl(const MulRequest* request, const std::vector<butil::IOBuf>& bufs) {
  auto field_type = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field_type);
  int64_t num_elements = request->num_elements();
  Shape shape = {1, num_elements};

  NdArrayRef u(element_type, shape);
  NdArrayRef v(element_type, shape);
  bufs.at(0).copy_to(u.data<void>());
  bufs.at(1).copy_to(v.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field_type, shape);
  NdArrayRef x = std::accumulate(r1.begin(), r1.end(), u, ring_add);

  std::vector<NdArrayRef> r2 = chip_.GenRand(field_type, shape);
  NdArrayRef y = std::accumulate(r2.begin(), r2.end(), v, ring_add);

  std::vector<NdArrayRef> r3 = chip_.GenRand(field_type, shape);

  return std::accumulate(r3.begin(), r3.end(), ring_mul(x, y), ring_sub);
}

NdArrayRef SquareImpl(const SquareRequest* request,
                      const std::vector<butil::IOBuf>& bufs) {
  auto field_type = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field_type);
  int64_t num_elements = request->num_elements();
  Shape shape = {1, num_elements};

  NdArrayRef u(element_type, shape);
  bufs.at(0).copy_to(u.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field_type, shape);
  NdArrayRef x = std::accumulate(r1.begin(), r1.end(), u, ring_add);

  std::vector<NdArrayRef> r2 = chip_.GenRand(field_type, shape);

  return std::accumulate(r2.begin(), r2.end(), ring_mul(x, x), ring_sub);
}

NdArrayRef DotImpl(const DotRequest* request,
                   const std::vector<butil::IOBuf>& bufs) {
  auto field_type = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field_type);
  int64_t m = request->m();
  int64_t n = request->n();
  int64_t k = request->k();

  NdArrayRef u(element_type, {m, k});
  NdArrayRef v(element_type, {k, n});
  bufs.at(0).copy_to(u.data<void>());
  bufs.at(1).copy_to(v.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field_type, {m, k});
  NdArrayRef x = std::accumulate(r1.begin(), r1.end(), u, ring_add);

  std::vector<NdArrayRef> r2 = chip_.GenRand(field_type, {k, n});
  NdArrayRef y = std::accumulate(r2.begin(), r2.end(), v, ring_add);

  std::vector<NdArrayRef> r3 = chip_.GenRand(field_type, {m, n});
  return std::accumulate(r3.begin(), r3.end(), ring_mmul(x, y), ring_sub);
}

NdArrayRef AndImpl(const AndRequest* request,
                   const std::vector<butil::IOBuf>& bufs) {
  auto field_type = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field_type);
  int64_t num_elements = request->num_elements();
  Shape shape = {1, num_elements};

  NdArrayRef u(element_type, shape);
  NdArrayRef v(element_type, shape);
  bufs.at(0).copy_to(u.data<void>());
  bufs.at(1).copy_to(v.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field_type, shape);
  NdArrayRef x = std::accumulate(r1.begin(), r1.end(), u, ring_xor);

  std::vector<NdArrayRef> r2 = chip_.GenRand(field_type, shape);
  NdArrayRef y = std::accumulate(r2.begin(), r2.end(), v, ring_xor);

  std::vector<NdArrayRef> r3 = chip_.GenRand(field_type, shape);

  return std::accumulate(r3.begin(), r3.end(), ring_and(x, y), ring_xor);
}

NdArrayRef TruncImpl(const TruncRequest* request,
                     const std::vector<butil::IOBuf>& bufs) {
  auto field_type = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field_type);
  int64_t num_elements = request->num_elements();
  Shape shape = {1, num_elements};

  NdArrayRef u(element_type, shape);
  bufs.at(0).copy_to(u.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field_type, shape);
  NdArrayRef x = std::accumulate(r1.begin(), r1.end(), u, ring_add);

  std::vector<NdArrayRef> r2 = chip_.GenRand(field_type, shape);

  return std::accumulate(
      r2.begin(), r2.end(),
      ring_arshift(x, {static_cast<int64_t>(request->bits())}), ring_sub);
}

BitSet EqzImpl(const EqzRequest* request,
               const std::vector<butil::IOBuf>& bufs) {
  auto field = static_cast<spu::FieldType>(request->field_type());
  auto element_type = makeType<RingTy>(field);
  int64_t num_elements = request->num_elements();
  Shape shape = {1, num_elements};

  NdArrayRef u(element_type, shape);
  bufs.at(0).copy_to(u.data<void>());

  std::vector<NdArrayRef> r1 = chip_.GenRand(field, shape);
  NdArrayRef z = std::accumulate(r1.begin(), r1.end(), u, ring_add);

  std::vector<BitSet> r2 = chip_.GenRand(num_elements);

  BitSet out{static_cast<size_t>(num_elements)};

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    NdArrayView<el_t> _z(z);

    pforeach(0, num_elements, [&](int64_t idx) {
      _z[idx] == 0 ? out.Set(idx, true) : out.Set(idx, false);
    });

    out = std::accumulate(r2.begin(), r2.end(), out, BitSet::Xor);
  });

  return out;
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
