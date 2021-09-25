/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Wei Kang)
 *
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

#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/python/csrc/k2.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"
#include "k2/python/csrc/torch/v2/ragged_arc.h"

using namespace k2;

TEST(RaggedArcTest, FromUnaryFunctionTensor) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 2 10
        0 1 1 20
        1 2 -1 30
        2)";

    RaggedArc src(s);
    src.SetRequiresGrad(true);

    src.SetAttr("float_attr",
                py::cast(torch::tensor(
                    {0.1, 0.2, 0.3},
                    torch::dtype(torch::kFloat32).requires_grad(true))));

    src.SetAttr("int_attr", py::cast(torch::tensor(
                                {1, 2, 3}, torch::dtype(torch::kInt32))));

    src.SetAttr("ragged_attr",
                py::cast(RaggedAny("[[1 2 3] [5 6] []]",
                                   py::cast(torch::kInt32), "cpu")));

    src.SetAttr("attr1", py::str("src"));
    src.SetAttr("attr2", py::str("fsa"));

    Array1<int32_t> arc_map;
    Ragged<Arc> arcs;
    ArcSort(src.fsa, &arcs, &arc_map);
    auto dest = RaggedArc(src, arcs, ToTorch<int32_t>(arc_map));

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(),
        torch::tensor({0.2, 0.1, 0.3}, torch::dtype(torch::kFloat32))));

    EXPECT_TRUE(torch::allclose(
        dest.Scores(),
        torch::tensor({20, 10, 30}, torch::dtype(torch::kFloat32))));

    EXPECT_TRUE(
        torch::equal(dest.GetAttr("int_attr").cast<torch::Tensor>(),
                     torch::tensor({2, 1, 3}, torch::dtype(torch::kInt32))));

    RaggedAny expected_ragged_attr =
        RaggedAny("[[5 6] [1 2 3] []]", py::cast(torch::kInt32), "cpu");
    EXPECT_EQ(dest.GetAttr("ragged_attr").cast<RaggedAny>().ToString(true),
              expected_ragged_attr.ToString(true));

    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              src.GetAttr("attr1").cast<std::string>());

    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              src.GetAttr("attr2").cast<std::string>());

    torch::Tensor scale =
        torch::tensor({10, 20, 30}, torch::dtype(torch::kFloat32));

    torch::Tensor sum_attr =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();

    torch::Tensor sum_score = (dest.Scores() * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({sum_attr}, {});
      torch::autograd::backward({sum_score}, {});
    }

    torch::Tensor expected_grad =
        torch::tensor({20, 10, 30}, torch::dtype(torch::kFloat32));

    EXPECT_TRUE(torch::allclose(
        src.GetAttr("float_attr").cast<torch::Tensor>().grad(), expected_grad));

    EXPECT_TRUE(torch::allclose(src.Scores().grad(), expected_grad));
  }
}

TEST(RaggedArcTest, FromUnaryFunctionRagged) {
  for (const ContextPtr &c : {GetCpuContext(), GetCudaContext()}) {
    std::string s = R"(0 1 0 0
        0 1 1 0
        1 2 -1 0
        2)";
    torch::Tensor scores = torch::tensor(
        {1, 2, 3}, torch::dtype(torch::kFloat32).requires_grad(true));
    torch::Tensor scores_copy = scores.detach().clone().requires_grad_(true);
    RaggedArc src(s);
    src.SetScores(scores);
    src.SetAttr("attr1", py::str("hello"));
    src.SetAttr("attr2", py::str("k2"));
    torch::Tensor float_attr = torch::tensor(
        {0.1, 0.2, 0.3}, torch::dtype(torch::kFloat32).requires_grad(true));
    src.SetAttr("float_attr",
                py::cast(float_attr.detach().clone().requires_grad_(true)));

    src.SetAttr("int_attr", py::cast(torch::tensor(
                                {1, 2, 3}, torch::dtype(torch::kInt32))));

    src.SetAttr("ragged_attr",
                py::cast(RaggedAny("[[10 20] [30 40 50] [60 70]]",
                                   py::cast(torch::kInt32), "cpu")));

    Ragged<int32_t> arc_map_raw;
    Ragged<Arc> arcs;
    RemoveEpsilon(src.fsa, src.Properties(), &arcs, &arc_map_raw);

    RaggedAny arc_map(arc_map_raw.Generic());

    RaggedArc dest = RaggedArc(src, arcs, arc_map);


    EXPECT_EQ(dest.GetAttr("attr1").cast<std::string>(),
              src.GetAttr("attr1").cast<std::string>());

    EXPECT_EQ(dest.GetAttr("attr2").cast<std::string>(),
              src.GetAttr("attr2").cast<std::string>());

    RaggedAny expected_arc_map =
        RaggedAny("[[1] [0 2] [2]]", py::cast(torch::kInt32), "cpu");

    EXPECT_EQ(arc_map.ToString(true), expected_arc_map.ToString(true));

    RaggedAny expected_int_attr =
        RaggedAny("[[2] [1 3] [3]]", py::cast(torch::kInt32), "cpu");
    EXPECT_EQ(dest.GetAttr("int_attr").cast<RaggedAny>().ToString(true),
              expected_int_attr.ToString(true));

    RaggedAny expected_ragged_attr = RaggedAny(
        "[[30 40 50] [10 20 60 70] [60 70]]", py::cast(torch::kInt32), "cpu");

    EXPECT_EQ(dest.GetAttr("ragged_attr").cast<RaggedAny>().ToString(true),
              expected_ragged_attr.ToString(true));

    torch::Tensor expected_float_attr =
        torch::empty_like(dest.GetAttr("float_attr").cast<torch::Tensor>());
    expected_float_attr[0] = float_attr[1];
    expected_float_attr[1] = float_attr[0] + float_attr[2];
    expected_float_attr[2] = float_attr[2];

    EXPECT_TRUE(torch::allclose(
        dest.GetAttr("float_attr").cast<torch::Tensor>(), expected_float_attr));

    torch::Tensor expected_scores = torch::empty_like(dest.Scores());
    expected_scores[0] = scores_copy[1];
    expected_scores[1] = scores_copy[0] + scores_copy[2];
    expected_scores[2] = scores_copy[2];

    EXPECT_TRUE(torch::allclose(dest.Scores(), expected_scores));

    torch::Tensor scale = torch::tensor({10, 20, 30}).to(float_attr);

    torch::Tensor float_attr_sum =
        (dest.GetAttr("float_attr").cast<torch::Tensor>() * scale).sum();

    torch::Tensor expected_float_attr_sum = (expected_float_attr * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({float_attr_sum}, {});
      torch::autograd::backward({expected_float_attr_sum}, {});
    }

    EXPECT_TRUE(
        torch::allclose(src.GetAttr("float_attr").cast<torch::Tensor>().grad(),
                        float_attr.grad()));

    torch::Tensor scores_sum = (dest.Scores() * scale).sum();
    torch::Tensor expected_scores_sum = (expected_scores * scale).sum();

    {
      py::gil_scoped_release no_gil;
      torch::autograd::backward({scores_sum}, {});
      torch::autograd::backward({expected_scores_sum}, {});
    }

    EXPECT_TRUE(torch::allclose(scores.grad(), scores_copy.grad()));
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  py::scoped_interpreter guard{};
  py::module_::import("torch");
  py::module_::import("_k2");
  return RUN_ALL_TESTS();
}
