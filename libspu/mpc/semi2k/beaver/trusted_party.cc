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

#include "libspu/mpc/semi2k/beaver/trusted_party.h"

#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {
namespace {

void checkDescs(absl::Span<const PrgArrayDesc> descs) {
  for (size_t idx = 1; idx < descs.size(); idx++) {
    SPU_ENFORCE(descs[0].field == descs[idx].field);
    SPU_ENFORCE(descs[0].shape == descs[idx].shape);
  }
}

}  // namespace

NdArrayRef TrustedParty::adjustMul(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);
  // adjust = rs[0] * rs[1] - rs[2];
  return ring_sub(ring_mul(rs[0], rs[1]), rs[2]);
}

NdArrayRef TrustedParty::adjustAnd(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(internal::RecOp::XOR, seeds, descs);
  // adjust = (rs[0] & rs[1]) ^ rs[2];
  return ring_xor(ring_and(rs[0], rs[1]), rs[2]);
}

NdArrayRef TrustedParty::adjustTrunc(Descs descs, Seeds seeds, size_t bits) {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  checkDescs(descs);

  auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);
  // adjust = (rs[0] >> bits) - rs[1];
  return ring_sub(ring_arshift(rs[0], bits), rs[1]);
}

std::pair<NdArrayRef, NdArrayRef> TrustedParty::adjustTruncPr(Descs descs,
                                                              Seeds seeds,
                                                              size_t bits) {
  // descs[0] is r, descs[1] adjust to r[k-2, bits], descs[2] adjust to r[k-1]
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);

  // adjust1 = ((rs[0] << 1) >> (bits + 1)) - rs[1];
  auto adjust1 = ring_sub(ring_rshift(ring_lshift(rs[0], 1), bits + 1), rs[1]);

  // adjust2 = (rs[0] >> (k - 1)) - rs[2];
  const size_t k = SizeOf(descs[0].field) * 8;
  auto adjust2 = ring_sub(ring_rshift(rs[0], k - 1), rs[2]);

  return {adjust1, adjust2};
}

NdArrayRef TrustedParty::adjustRandBit(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 1U);
  auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);

  // adjust = bitrev - rs[0];
  return ring_sub(ring_randbit(descs[0].field, descs[0].shape), rs[0]);
}

NdArrayRef TrustedParty::adjustEqz(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  checkDescs(descs);
  auto rs_a = reconstruct(internal::RecOp::ADD, seeds, descs.subspan(0, 1));
  auto rs_b = reconstruct(internal::RecOp::XOR, seeds, descs.subspan(1, 2));
  // adjust = rs[0] ^ rs[1];
  return ring_xor(rs_a[0], rs_b[0]);
}

NdArrayRef TrustedParty::adjustPerm(Descs descs, Seeds seeds,
                                    absl::Span<const int64_t> perm_vec) {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);

  return ring_sub(applyInvPerm(rs[0], perm_vec), rs[1]);
}

}  // namespace spu::mpc::semi2k
