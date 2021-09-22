/**
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                2021  Xiaomi Corp.       (author: Daniel Povey,
 *                                                  Haowen Qiu,
 *                                                  Wei Kang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "k2/csrc/context.h"
#include "k2/csrc/device_guard.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/python/csrc/torch/v2/ops.h"

namespace k2 {

  void IndexAdd(torch::Tensor index, torch::Tensor value,
                torch::Tensor *in_out) {
    NVTX_RANGE(K2_FUNC);
    DeviceGuard guard(GetContext(index));

    Array1<int32_t> indexes = FromTorch<int32_t>(index);
    Tensor src = FromTorch(value, TensorTag{});
    Tensor dest = FromTorch(*in_out, TensorTag{});
    IndexAdd(src, indexes, true, &dest);
  }

  template <typename T>
  torch::Tensor IndexSelect1D(torch::Tensor src, torch::Tensor index,
                              T default_value) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
    K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value)
        << "Expected equal type"
        << " Given : " << src.scalar_type() << ", " << ToScalarType<T>::value;

    K2_CHECK_EQ(index.dim(), 1)
        << "Expected index dim: 1. Given : " << index.dim();
    K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value)
        << "Expected type int32_t Given : " << index.scalar_type();
    K2_CHECK(index.is_contiguous()) << "Expected contiguous";
    K2_CHECK_EQ(src.device(), index.device())
        << "Expected in the same device"
        << " Given : " << src.device() << ", " << index.device();

    bool allow_minus_one = true;
    Array1<int32_t> index_array = FromTorch<int32_t>(index);
    // If index_array.Dim() equals to zero, the `Index` below would produce an
    // ans with `ans.Data()` be a nullptr, which will cause crash when calling
    // `torch::from_blob`. Just return an empty tensor here.
    // If src is an empty tensor, we should return an empty torch.
    if (index_array.Dim() == 0 || src.numel() == 0)
      return torch::empty({0}, src.options());
    if (src.is_contiguous()) {
      Array1<T> src_array = FromTorch<T>(src);
      Array1<T> ans_array =
          Index(src_array, index_array, allow_minus_one, default_value);
      return ToTorch(ans_array);
    }
    Tensor tensor = FromTorch(src, TensorTag{});
    Tensor ans = Index(tensor, index_array, allow_minus_one, default_value);
    return ToTorch(ans);
  }

  template <typename T>
  torch::Tensor IndexSelect2D(torch::Tensor src, torch::Tensor index) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_EQ(src.dim(), 2) << "Expected dim: 2. Given: " << src.dim();
    K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);

    K2_CHECK_EQ(index.dim(), 1);
    K2_CHECK_EQ(index.scalar_type(), ToScalarType<int32_t>::value);
    K2_CHECK(index.is_contiguous());
    K2_CHECK_EQ(src.device(), index.device());

    Array2<T> src_array = FromTorch<T>(src, Array2Tag{});
    Array1<int32_t> index_array = FromTorch<int32_t>(index);
    // If index_array.Dim() equals to zero, the `IndexRows` below would produce
    // an ans with `ans.Data()` be a nullptr, which will cause crash when
    // calling `torch::from_blob`. Just return an empty tensor here.
    // If src is an empty tensor, we should return an empty torch.
    if (index_array.Dim() == 0 || src.sizes()[0] == 0)
      return torch::empty({0, src.sizes()[1]}, src.options());
    bool allow_minus_one = true;
    Array2<T> ans_array = IndexRows(src_array, index_array, allow_minus_one);

    return ToTorch(ans_array);
  }

  torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index,
                            double default_value /*= 0*/) {
    NVTX_RANGE(K2_FUNC);
    DeviceGuard guard(GetContext(src));
    auto scalar_type = src.scalar_type();
    if (src.dim() == 1) {
      switch (scalar_type) {
        case ToScalarType<int32_t>::value: {
          int32_t i = static_cast<int32_t>(default_value);
          K2_CHECK_EQ(static_cast<double>(i), default_value);
          return IndexSelect1D<int32_t>(src, index, i);
        }
        case ToScalarType<int64_t>::value: {
          int64_t i = static_cast<int64_t>(default_value);
          K2_CHECK_EQ(static_cast<double>(i), default_value);
          return IndexSelect1D<int64_t>(src, index, i);
        }
        case ToScalarType<float>::value:
          return IndexSelect1D<float>(src, index, default_value);
        case ToScalarType<double>::value:
          return IndexSelect1D<double>(src, index, default_value);
        default:
          K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
          return {};
      }
    } else if (src.dim() == 2) {
      switch (scalar_type) {
        case ToScalarType<int32_t>::value:
          return IndexSelect2D<int32_t>(src, index);
        case ToScalarType<int64_t>::value:
          return IndexSelect2D<int64_t>(src, index);
        case ToScalarType<float>::value:
          return IndexSelect2D<float>(src, index);
        case ToScalarType<double>::value:
          return IndexSelect2D<double>(src, index);
        default:
          K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
          return {};
      }
    } else {
      K2_LOG(FATAL) << "Unsupported dim: " << src.dim()
                    << ".\nIt supports only 1-D and 2-D tensors.";
      return {};
    }
  }

  template <typename T>
  torch::Tensor SimpleRaggedIndexSelect1D(torch::Tensor src,
                                          Ragged<int32_t> &indexes) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_EQ(src.dim(), 1) << "Expected dim: 1. Given: " << src.dim();
    K2_CHECK_EQ(src.scalar_type(), ToScalarType<T>::value);
    K2_CHECK_EQ(indexes.NumAxes(), 2);
    ContextPtr context = GetContext(src);
    K2_CHECK(context->IsCompatible(*indexes.Context()));

    Tensor tensor = FromTorch(src, TensorTag{});
    Tensor ans = SimpleRaggedIndexSelect1D(tensor, indexes);
    return ToTorch(ans);
  }

  torch::Tensor SimpleRaggedIndexSelect(torch::Tensor src,
                                        RaggedAny &ragged) {
    DeviceGuard guard(GetContext(src));
    Ragged<int32_t> indexes = ragged.any.Specialize<int32_t>();
    auto scalar_type = src.scalar_type();
    if (src.dim() == 1) {
      switch (scalar_type) {
        case ToScalarType<int32_t>::value:
          return SimpleRaggedIndexSelect1D<int32_t>(src, indexes);
        case ToScalarType<float>::value:
          return SimpleRaggedIndexSelect1D<float>(src, indexes);
        default:
          K2_LOG(FATAL) << "Unsupported scalar type: " << scalar_type;
          return {};
      }
    } else {
      K2_LOG(FATAL) << "Unsupported dim: " << src.dim()
                    << ". It supports only 1-D tensors for now";
      return {};
    }
  }

}  // namespace k2
