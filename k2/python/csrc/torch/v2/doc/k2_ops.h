/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_DOC_K2_OPS_H_
#define K2_PYTHON_CSRC_TORCH_V2_DOC_K2_OPS_H_

namespace k2 {

static constexpr const char *kTensorIndexSelectDoc = R"doc(
Create a ragged tensor with arbitrary number of axes.

Note:
  A ragged tensor has at least two axes.

Hint:
  The returned tensor is on CPU.

>>> import torch
>>> import k2.ragged as k2r
>>> a = k2r.create_ragged_tensor([ [1, 2], [5], [], [9] ])
>>> a
RaggedTensor([[1, 2],
              [5],
              [],
              [9]], dtype=torch.int32)
>>> a.dtype
torch.int32
>>> b = k2r.create_ragged_tensor([ [1, 3.0], [] ])
>>> b
RaggedTensor([[1, 3],
              []], dtype=torch.float32)
>>> b.dtype
torch.float32
>>> c = k2r.create_ragged_tensor([ [1] ], dtype=torch.float64)
>>> c.dtype
torch.float64
>>> d = k2r.create_ragged_tensor([ [[1], [2, 3]], [[4], []] ])
>>> d
RaggedTensor([[[1],
               [2, 3]],
              [[4],
               []]], dtype=torch.int32)
>>> d.num_axes
3
>>> e = k2r.create_ragged_tensor([])
>>> e
RaggedTensor([], dtype=torch.int32)
>>> e.num_axes
2
>>> e.shape.row_splits(1)
tensor([0], dtype=torch.int32)
>>> e.shape.row_ids(1)
tensor([], dtype=torch.int32)
>>> f = k2r.create_ragged_tensor([ [1, 2], [], [3] ], device=torch.device('cuda', 0))
>>> f
RaggedTensor([[1, 2],
              [],
              [3]], device='cuda:0', dtype=torch.int32)
>>> e = k2r.create_ragged_tensor([[1], []], device='cuda:1')
>>> e
RaggedTensor([[1],
              []], device='cuda:1', dtype=torch.int32)

Args:
  data:
    A list-of sublist(s) of integers or real numbers.
    It can have arbitrary number of axes (at least two).
  dtype:
    Optional. If None, it infers the dtype from ``data``
    automatically, which is either ``torch.int32`` or
    ``torch.float32``. Supported dtypes are: ``torch.int32``,
    ``torch.float32``, and ``torch.float64``.
  device:
    It can be either an instance of ``torch.device`` or
    a string representing a torch device. Example
    values are: ``"cpu"``, ``"cuda:0"``, ``torch.device("cpu")``,
    ``torch.device("cuda", 0)``.

Returns:
  Return a ragged tensor.
)doc";

static constexpr const char *kSimpleRaggedIndexSelectDoc = R"doc(

)doc";

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_DOC_K2_OPS_H_


