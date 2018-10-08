//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_with_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 5};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 2};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}, {1, 5, 7, 5, 6}, {0, 6, 2, 10, 2}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 0}, {0, 0, 8, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, 0, 1, 0}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 6};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 1}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// Adapted from the XLA docs to provide an example in >2D: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_3d_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       op::ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{2, 4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 4, 6};
    Shape window_shape{2, 2, 3};
    auto window_strides = Strides{2, 2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        test::NDArray<float, 3>(
            {{{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}},
             {{2, 5, 8, 3, 4, 2}, {1, 2, 8, 4, 5, 2}, {10, 2, 3, 4, 1, 0}, {4, 1, 2, 4, 5, 7}}})
            .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 3>({{{2, 6}, {3, 1}}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(
        (test::NDArray<float, 3>(
             {{{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
              {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}})
             .get_vector()),
        read_vector<float>(result));
}
