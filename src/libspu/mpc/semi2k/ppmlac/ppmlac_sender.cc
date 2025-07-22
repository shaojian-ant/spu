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

#include "libspu/mpc/semi2k/ppmlac/ppmlac_sender.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k::ppmlac {

void SenderPPMLAC::Initialize() {
  // Recv receiver's public key
  yacl::Buffer recv_pk = lctx_->Recv(recv_rank_, "PP_MLAC:pub_key");
  SPDLOG_DEBUG("Recv public key from rank {}", recv_rank_);

  // Send encrypted public key to receiver
  yacl::Buffer enc_pk = chip_.EncryptPubKey(recv_pk);
  lctx_->SendAsync(recv_rank_, enc_pk, "PPMLAC:pub_key");
  SPDLOG_DEBUG("Send encrypted public key to rank {}", recv_rank_);

  // Send encrypted random number to receiver
  yacl::Buffer enc_rn = chip_.EncryptRandNum(recv_pk);
  lctx_->SendAsync(recv_rank_, enc_rn, "PPMLAC:rand_num");
  SPDLOG_DEBUG("Send encrypted random number to rank {}", recv_rank_);

  // Recv encrypted random number from receiver
  yacl::Buffer recv_enc_rn = lctx_->Recv(recv_rank_, "PPMLAC:rand_num");
  SPDLOG_DEBUG("Recv encrypted random number from rank {}", recv_rank_);

  chip_.SetupPRNG(recv_rank_, recv_enc_rn);

  SPDLOG_DEBUG("PPMLAC finished initialization with rank {}", lctx_->Rank());
}

NdArrayRef SenderPPMLAC::Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                             const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  const Type& eltype = x.eltype();
  const NdArrayRef a = chip_.GenRand(recv_rank_, eltype, x.shape());  // r_1
  const NdArrayRef b = chip_.GenRand(recv_rank_, eltype, y.shape());  // r_2
  const NdArrayRef q = chip_.GenRand(recv_rank_, eltype, x.shape());  // q_1

  vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, recv_rank_, "send(x-a,y-b)");
  });

  return q;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::Square(KernelEvalContext* ctx, const NdArrayRef& x) {
  auto* comm = ctx->getState<Communicator>();
  const NdArrayRef a = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // r_1
  const NdArrayRef q = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // q_1

  comm->reduce(ReduceOp::ADD, ring_sub(x, a), recv_rank_, "send(x-a)");

  return q;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::MatMul(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  const Type& eltype = x.eltype();
  const NdArrayRef a = chip_.GenRand(recv_rank_, eltype, x.shape());  // r_1
  const NdArrayRef b = chip_.GenRand(recv_rank_, eltype, y.shape());  // r_2
  const NdArrayRef q = chip_.GenRand(
      recv_rank_, eltype, {x.shape().at(0), y.shape().at(1)});  // q_1

  vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, recv_rank_, "send(x-a,y-b)");
  });

  return q;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::And(KernelEvalContext* ctx, const NdArrayRef& lhs,
                             const NdArrayRef& rhs) {
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  SPU_ENFORCE(lhs.eltype().as<Ring2k>()->field() ==
              rhs.eltype().as<Ring2k>()->field());

  auto* comm = ctx->getState<Communicator>();
  const FieldType field = lhs.eltype().as<Ring2k>()->field();

  const size_t out_nbits = std::min(lhs.eltype().as<BShrTy>()->nbits(),
                                    rhs.eltype().as<BShrTy>()->nbits());

  const PtType backtype = getBacktype(out_nbits);
  const int64_t numel = lhs.numel();

  const Type eltype = makeType<RingTy>(FM128);
  int64_t num128 = CeilDiv(numel * SizeOf(backtype), SizeOf(FM128));
  NdArrayRef a = chip_.GenRand(recv_rank_, eltype, {num128});  // r_1
  NdArrayRef b = chip_.GenRand(recv_rank_, eltype, {num128});  // r_2
  NdArrayRef q = chip_.GenRand(recv_rank_, eltype, {num128});  // q_1

  NdArrayRef out(makeType<BShrTy>(field, out_nbits), lhs.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _lhs(lhs);
    NdArrayView<T> _rhs(rhs);

    DISPATCH_UINT_PT_TYPES(backtype, [&]() {
      using V = ScalarT;

      absl::Span<const V> _a(a.data<V>(), numel);
      absl::Span<const V> _b(b.data<V>(), numel);
      absl::Span<const V> _q(q.data<V>(), numel);

      // first half mask x^a, second half mask y^b.
      std::vector<V> mask(numel * 2, 0);
      pforeach(0, numel, [&](int64_t idx) {
        mask[idx] = _lhs[idx] ^ _a[idx];
        mask[numel + idx] = _rhs[idx] ^ _b[idx];
      });

      mask = comm->reduce<V, std::bit_xor>(mask, recv_rank_, "open(x^a,y^b)");

      NdArrayView<T> _out(out);
      pforeach(0, numel, [&](int64_t idx) { _out[idx] = _q[idx]; });
    });
  });

  return out;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::Trunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               size_t bits) {
  auto* comm = ctx->getState<Communicator>();
  const NdArrayRef a = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // r_1
  const NdArrayRef q = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // q_1

  comm->reduce(ReduceOp::ADD, ring_sub(x, a), recv_rank_, "send(x-a)");

  return q;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::B2A(KernelEvalContext* ctx, const NdArrayRef& x) {
  auto* comm = ctx->getState<Communicator>();
  const FieldType field = x.eltype().as<Ring2k>()->field();
  const size_t out_nbits = x.eltype().as<BShrTy>()->nbits();
  const PtType backtype = getBacktype(out_nbits);
  const int64_t numel = x.numel();

  int64_t num128 = CeilDiv(numel * SizeOf(backtype), SizeOf(FM128));
  NdArrayRef a = chip_.GenRand(recv_rank_, makeType<RingTy>(FM128), {num128});
  NdArrayRef q = chip_.GenRand(recv_rank_, makeType<AShrTy>(field), x.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _x(x);

    DISPATCH_UINT_PT_TYPES(backtype, [&]() {
      using V = ScalarT;
      absl::Span<const V> _a(a.data<V>(), numel);

      std::vector<V> mask(numel, 0);
      pforeach(0, numel, [&](int64_t idx) { mask[idx] = _x[idx] ^ _a[idx]; });

      mask = comm->reduce<V, std::bit_xor>(mask, recv_rank_, "open(x^a)");
    });
  });

  return q;  // [z]_0 = q_1
}

NdArrayRef SenderPPMLAC::Eqz(KernelEvalContext* ctx, const NdArrayRef& z) {
  const FieldType field = z.eltype().as<Ring2k>()->field();
  const NdArrayRef a = chip_.GenRand(recv_rank_, z.eltype(), z.shape());  // r_1
  const BitSet q = chip_.GenRandBits(recv_rank_, z.numel());              // q_1

  auto* comm = ctx->getState<Communicator>();
  comm->reduce(ReduceOp::ADD, ring_sub(z, a), recv_rank_, "send(z-a)");

  NdArrayRef out(makeType<BShrTy>(field), z.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    // 1 bit info in lsb
    NdArrayView<el_t> _out(out);
    pforeach(0, out.numel(), [&](int64_t idx) { _out[idx] = q.At(idx); });
  });

  return out;
}

NdArrayRef SenderPPMLAC::Perm(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& perm, size_t perm_rank) {
  auto* comm = ctx->getState<Communicator>();
  const NdArrayRef a = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // r_1
  const NdArrayRef q = chip_.GenRand(recv_rank_, x.eltype(), x.shape());  // q_1

  comm->reduce(ReduceOp::ADD, ring_sub(x, a), recv_rank_, "send(x-a)");

  if (perm_rank == lctx_->Rank()) {
    const NdArrayRef rand_perm = chip_.GenRandPerm(recv_rank_, perm.shape());
    const NdArrayRef perm_mask = applyPerm(perm, rand_perm);
    comm->sendAsync(perm_rank, perm_mask, "send(perm_mask)");
  }

  return q;  // [z]_0 = q_1
}

}  // namespace spu::mpc::semi2k::ppmlac
