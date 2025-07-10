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

#include "libspu/core/prelude.h"
#include "libspu/mpc/semi2k/ppmlac/utils.h"

namespace spu::mpc::semi2k::ppmlac {

class BitSet : std::vector<uint8_t> {
  using Base = std::vector<uint8_t>;

 public:
  explicit BitSet(size_t size)
      : std::vector<uint8_t>(CeilDiv(size, 8)), size_(size) {}

  uint8_t* Data() { return data(); }

  const uint8_t* Data() const { return data(); }

  size_t Size() const { return size_; }

  size_t SizeInBytes() const { return size(); }

  bool At(std::size_t pos) const {
    SPU_ENFORCE(pos < size_);
    return at(pos / 8) & (1 << (pos % 8));
  }

  BitSet& Set(std::size_t pos, bool value = true) {
    SPU_ENFORCE(pos < size_);

    uint8_t& byte = (*this)[pos / 8];
    const uint8_t mask = 1 << (pos % 8);

    if (value) {
      byte |= mask;
    } else {
      byte &= ~mask;
    }

    return *this;
  }

  static BitSet Xor(const BitSet& lhs, const BitSet& rhs) {
    SPU_ENFORCE(lhs.Size() == rhs.Size());
    BitSet result(lhs.Size());
    for (size_t i = 0; i < lhs.size(); ++i) {
      result[i] = lhs.at(i) ^ rhs.at(i);
    }
    return result;
  }

 private:
  size_t size_;
};

}  // namespace spu::mpc::semi2k::ppmlac
