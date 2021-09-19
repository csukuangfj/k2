/**
 * @brief Index select for k2.
 *
 * Unlike torch.index_select, when an entry is -1, it sets
 * the destination entry to 0.
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corp.       (author: Daniel Povey, Haowen Qiu)
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

#include "k2/python/csrc/torch/index_select.h"
#include "k2/python/csrc/torch/v2/ops.h"

static void IndexSelect(py::module &m) {
  m.def("index_select", &k2::IndexSelect, py::arg("src"), py::arg("index"),
        py::arg("default_value") = 0,
        R"(
      Args:
        src:
          It can be either a 1-D or a 2-D tensor. Supported dtypes are:
          `torch.int32`, `torch.int64`, `torch.float32`, and `torch.float64`.
        index:
          It has to be a 1-D **contiguous** tensor with dtype `torch.int32`.
          Must satisfy `-1 <= index[i] < src.shape[0]`.
        default_value:
          It is the default value for ans[i] if index[i] is -1.
          Used only when `src` is a 1-D tensor.
      Returns:
        Return a tensor:
          - `ans.ndim == src.ndim`
          - `ans.shape[0] == index.shape[0]`
          - If `ans.ndim == 2`, then `ans.shape[1] == src.shape[1]`
          - `ans[i] = src[index[i]]` if `index[i] != -1`.
          - `ans[i] = default_value` if `index[i] == -1`
      )");
  m.def("simple_ragged_index_select", &k2::SimpleRaggedIndexSelect,
        py::arg("src"), py::arg("indexes"));
}

void PybindIndexSelect(py::module &m) { IndexSelect(m); }
