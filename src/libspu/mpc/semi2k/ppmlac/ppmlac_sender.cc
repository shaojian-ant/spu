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
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k::ppmlac {
void SenderPPMLAC::Initialize() {
  // Recv receiver's public key
  auto recv_pk = lctx_->Recv(recv_rank_, "PP_MLAC:pub_key");
  SPDLOG_INFO("Recv public key from rank {}", recv_rank_);

  // Send encrypted public key to receiver
  auto enc_pk = chip_.EncryptPubKey(recv_pk);
  lctx_->SendAsync(recv_rank_, enc_pk, "PPMLAC:pub_key");
  SPDLOG_INFO("Send encrypted public key to rank {}", recv_rank_);

  // Send encrypted random number to receiver
  auto enc_rn = chip_.EncryptRandNum(recv_pk);
  lctx_->SendAsync(recv_rank_, enc_rn, "PPMLAC:rand_num");
  SPDLOG_INFO("Send encrypted random number to rank {}", recv_rank_);

  // Recv encrypted random number from receiver
  auto recv_enc_rn = lctx_->Recv(recv_rank_, "PPMLAC:rand_num");
  SPDLOG_INFO("Recv encrypted random number from rank {}", recv_rank_);

  chip_.SetupPRNG(recv_rank_, recv_enc_rn);
}

NdArrayRef SenderPPMLAC::Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                             const NdArrayRef& y) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = x.eltype().as<Ring2k>()->field();
  const auto a = chip_.GenRnd(recv_rank_, field, x.shape());  // r_1
  const auto b = chip_.GenRnd(recv_rank_, field, y.shape());  // r_2
  const auto q = chip_.GenRnd(recv_rank_, field, x.shape());  // q_1

  vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->reduce(ReduceOp::ADD, s, recv_rank_, "send(x-a,y-b)");
  });

  return q;  // [z]_0 = q_1
}

}  // namespace spu::mpc::semi2k::ppmlac
