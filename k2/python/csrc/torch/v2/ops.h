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

#ifndef K2_PYTHON_CSRC_TORCH_V2_OPS_H_
#define K2_PYTHON_CSRC_TORCH_V2_OPS_H_

#include <utility>

#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"

namespace k2 {

  void IndexAdd(torch::Tensor index, torch::Tensor value,
                torch::Tensor *in_out);

  /* Returns a 1-D tensor which indexes the src tensor using entries
     from `index`.

     @param  [in]  src    A 1-D tensor.
     @param  [in]  index  A 1-D tensor with dtype torch.int32.
                          It has to satisfy:
                              -1 <= index[i] < src.numel()
                              for i in [0, index.numel())
                          CAUTION: We require that index.is_contiguous()
                                   is true.
     @param [in] default_value  The value for ans[i] when index[i] is -1.
     @return
        Returns a 1-D contiguous tensor such that:
            ans[i] = src[index[i]] if index[i] > 0
            ans[i] = default_value if index[i] is -1
   */
  template <typename T>
  torch::Tensor IndexSelect1D(torch::Tensor src, torch::Tensor index,
                              T default_value);

  /* Returns a 2-D tensor which indexes the src tensor using entries
     from `index`.

     @param  [in]  src    A 2-D tensor. If it is non-contiguous, then it
                          has to satisfy src.strides()[1] == 1.

     @param  [in]  index  A 1-D tensor with dtype torch.int32.
                          It has to satisfy:
                              -1 <= index[i] < src.shape()[0]
                              for i in [0, index.numel())
                          CAUTION: We require that index.is_contiguous()
                                   is true.
     @return
        Returns a 2-D contiguous tensor such that:
            ans[i] = src[index[i]] if index[i] > 0
            ans[i] = zero tensor whose numel() is src.shape()[1],
                     if index[i] is -1
   */
  template <typename T>
  torch::Tensor IndexSelect2D(torch::Tensor src, torch::Tensor index);

  torch::Tensor IndexSelect(torch::Tensor src, torch::Tensor index,
                            double default_value = 0);

  /*
    Returns a 1-D Tensor that is a result of indexing 1-D `src` with Ragged
    array `indexes` whose NumAxes() is 2. ans.numel() will equal to
    indexes.Dim0() as we suppose there is at most one non-zero element in `src`
    for any indexes sub-list in `indexes`.

       @param [in] src  Source tensor, to be indexed.
       @param [in] indexes   Indexes to use whose NumAxes() == 2, for any
                        sub-list `i` in `indexes`, we suppose there is at most
                        one non-zero values in `src` and we'll set ans[i]
                        with that non-zero value; if all values for
                        sub-list `i` is zero or the sub-list is empty, we just
                        set ans[i] == 0.
       @return   Returns a Tensor with the same dtype as `src` and shape
                       (indexes.Dim0()), i.e. a 1-D tensor with numel() equal
                       to `indexes.Dim0()`.
                       Noted the ans would be contiguous even though `src`
                       is not contiguous.
   */
  template <typename T>
  torch::Tensor SimpleRaggedIndexSelect1D(torch::Tensor src,
                                          Ragged<int32_t> &indexes);

  torch::Tensor SimpleRaggedIndexSelect(torch::Tensor src,
                                        RaggedAny &ragged);
}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_OPS_H_
