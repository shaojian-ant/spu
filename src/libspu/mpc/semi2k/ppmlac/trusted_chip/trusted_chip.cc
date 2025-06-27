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

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_chip.h"

#include "yacl/crypto/pke/sm2_enc.h"
#include "yacl/crypto/rand/rand.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::semi2k::ppmlac {

TrustedChip::TrustedChip(const std::string &asym_crypto_schema,
                         yacl::ByteContainerView public_key,
                         yacl::ByteContainerView private_key)
    : asym_crypto_schema_(asym_crypto_schema),
      public_key_(public_key),
      private_key_(private_key) {
  rand_num_ = yacl::crypto::SecureRandU128();
}

yacl::Buffer TrustedChip::EncryptPubKey(yacl::ByteContainerView pub_key) const {
  yacl::crypto::Sm2Encryptor encryptor(pub_key);
  auto enc_pk = encryptor.Encrypt(public_key_);
  return {enc_pk.data(), enc_pk.size()};
}

yacl::Buffer TrustedChip::EncryptRandNum(
    yacl::ByteContainerView pub_key) const {
  yacl::crypto::Sm2Encryptor encryptor(pub_key);
  auto enc_rn = encryptor.Encrypt({&rand_num_, sizeof(rand_num_)});
  return {enc_rn.data(), enc_rn.size()};
}

yacl::Buffer TrustedChip::DecryptPubKey(yacl::ByteContainerView enc_pk) const {
  yacl::crypto::Sm2Decryptor decryptor(private_key_);
  std::vector<uint8_t> pk_vec = decryptor.Decrypt(enc_pk);
  return {pk_vec.data(), pk_vec.size()};
}

uint128_t TrustedChip::DecryptRandNum(yacl::ByteContainerView enc_rn) const {
  yacl::crypto::Sm2Decryptor decryptor(private_key_);
  std::vector<uint8_t> rn_vec = decryptor.Decrypt(enc_rn);
  SPU_ENFORCE(rn_vec.size() == sizeof(uint128_t));
  uint128_t rand_num;
  memcpy(&rand_num, rn_vec.data(), sizeof(uint128_t));
  return rand_num;
}

void TrustedChip::SetupPRNG(size_t rank, yacl::ByteContainerView enc_rn) {
  const uint128_t rand_num = DecryptRandNum(enc_rn);
  const uint128_t seed = rand_num + rand_num_;
  prngs_.emplace(rank, seed);
}

}  // namespace spu::mpc::semi2k::ppmlac