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

#include <utility>

#include "absl/types/span.h"

#include "libspu/mpc/common/drbg_tensor.h"
#include "libspu/mpc/common/prg_tensor.h"

namespace spu::mpc::semi2k {

namespace internal {

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

template <typename T>
std::vector<NdArrayRef> reconstruct(RecOp op, absl::Span<const T> seeds,
                                    absl::Span<const PrgArrayDesc> descs) {
  std::vector<NdArrayRef> rs(descs.size());

  for (size_t rank = 0; rank < seeds.size(); ++rank) {
    for (size_t idx = 0; idx < descs.size(); ++idx) {
      // FIXME: TTP adjuster server and client MUST have same endianness.
      NdArrayRef t;
      if constexpr (std::is_convertible_v<T, PrgSeed>) {
        t = prgReplayArray(seeds[rank], descs[idx]);
      } else if constexpr (std::is_convertible_v<T, DrbgPtr>) {
        t = DrbgReplayArray(*seeds[rank], descs[idx]);
      } else {
        SPU_THROW("Unexpected exception");
      }

      if (rank == 0) {
        rs[idx] = t;
      } else {
        if (op == RecOp::ADD) {
          ring_add_(rs[idx], t);
        } else if (op == RecOp::XOR) {
          ring_xor_(rs[idx], t);
        } else {
          SPU_ENFORCE("not supported reconstruct op");
        }
      }
    }
  }

  return rs;
}

}  // namespace internal

class TrustedParty {
 private:
  using Seeds = absl::Span<const PrgSeed>;
  using Descs = absl::Span<const PrgArrayDesc>;

 public:
  static NdArrayRef adjustMul(Descs descs, Seeds seeds);

  template <typename T>
  static NdArrayRef adjustDot(Descs descs, absl::Span<const T> seeds, int64_t m,
                              int64_t n, int64_t k) {
    SPU_ENFORCE_EQ(descs.size(), 3U);
    SPU_ENFORCE_EQ(descs[0].shape, (std::vector<int64_t>{m, k}));
    SPU_ENFORCE_EQ(descs[1].shape, (std::vector<int64_t>{k, n}));
    SPU_ENFORCE_EQ(descs[2].shape, (std::vector<int64_t>{m, n}));

    auto rs = reconstruct(internal::RecOp::ADD, seeds, descs);
    // adjust = rs[0] dot rs[1] - rs[2];
    return ring_sub(ring_mmul(rs[0], rs[1]), rs[2]);
  }

  static NdArrayRef adjustAnd(Descs descs, Seeds seeds);

  static NdArrayRef adjustTrunc(Descs descs, Seeds seeds, size_t bits);

  static std::pair<NdArrayRef, NdArrayRef> adjustTruncPr(Descs descs,
                                                         Seeds seeds,
                                                         size_t bits);

  static NdArrayRef adjustRandBit(Descs descs, Seeds seeds);

  static NdArrayRef adjustEqz(Descs descs, Seeds seeds);

  static NdArrayRef adjustPerm(Descs descs, Seeds seeds,
                               absl::Span<const int64_t> perm_vec);
};

}  // namespace spu::mpc::semi2k
