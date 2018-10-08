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

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image)
{
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image)
{
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom}},
                                                           {{3 / denom,
                                                             4 / denom,
                                                             2 / denom,
                                                             1 / denom,
                                                             0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_2channel_2image)
{
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom},
                                                            {0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             4 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             1 / denom}},

                                                           {{3 / denom,
                                                             4 / denom,
                                                             2 / denom,
                                                             1 / denom,
                                                             0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom},
                                                            {3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom,
                                                             1 / denom,
                                                             2 / denom,
                                                             4 / denom,
                                                             3 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f =
        make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call_with_validate(f, {result}, {a});

    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>({{{{6 / denom, 8 / denom, 5 / denom}, // img 0 chan 0
                                   {7 / denom, 5 / denom, 3 / denom},
                                   {5 / denom, 2 / denom, 5 / denom},
                                   {6 / denom, 5 / denom, 5 / denom}},

                                  {{5 / denom, 7 / denom, 6 / denom}, // img 0 chan 1
                                   {8 / denom, 6 / denom, 7 / denom},
                                   {7 / denom, 2 / denom, 3 / denom},
                                   {6 / denom, 1 / denom, 0 / denom}}},

                                 {{{5 / denom, 6 / denom, 5 / denom}, // img 1 chan 0
                                   {3 / denom, 5 / denom, 9 / denom},
                                   {3 / denom, 6 / denom, 9 / denom},
                                   {2 / denom, 3 / denom, 3 / denom}},

                                  {{5 / denom, 3 / denom, 1 / denom}, // img 1 chan 1
                                   {6 / denom, 5 / denom, 4 / denom},
                                   {7 / denom, 5 / denom, 6 / denom},
                                   {4 / denom, 2 / denom, 4 / denom}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_strided)
{
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(A, window_shape, window_movement_strides), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                                             {6 / denom, 5 / denom, 8 / denom},
                                                             {6 / denom, 2 / denom, 4 / denom}}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded)
{
    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                  {{3.0f / 1, 8.0f / 2, 7.0f / 2, 2.0f / 1},
                                                   {5.0f / 2, 10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                   {5.0f / 2, 11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                   {3.0f / 1, 9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_below)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2},
                                                           {0.0f / 2, 4.0f / 4, 6.0f / 4},
                                                           {2.0f / 2, 5.0f / 4, 5.0f / 4}},
                                                          {{3.0f / 1, 8.0f / 2, 7.0f / 2},
                                                           {5.0f / 2, 10.0f / 4, 16.0f / 4},
                                                           {5.0f / 2, 11.0f / 4, 20.0f / 4}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_above)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                           {5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                           {2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                          {{10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                           {11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                           {9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 2, 0.0f / 1},
                                   {0.0f / 2, 4.0f / 4, 6.0f / 6, 6.0f / 4, 2.0f / 2},
                                   {2.0f / 3, 6.0f / 6, 8.0f / 9, 6.0f / 6, 2.0f / 3},
                                   {2.0f / 2, 5.0f / 4, 7.0f / 6, 5.0f / 4, 2.0f / 2},
                                   {2.0f / 1, 2.0f / 2, 2.0f / 3, 0.0f / 2, 0.0f / 1}},
                                  {{3.0f / 1, 8.0f / 2, 10.0f / 3, 7.0f / 2, 2.0f / 1},
                                   {5.0f / 2, 10.0f / 4, 21.0f / 6, 16.0f / 4, 11.0f / 2},
                                   {8.0f / 3, 19.0f / 6, 35.0f / 9, 27.0f / 6, 16.0f / 3},
                                   {5.0f / 2, 11.0f / 4, 25.0f / 6, 20.0f / 4, 14.0f / 2},
                                   {3.0f / 1, 9.0f / 2, 14.0f / 3, 11.0f / 2, 5.0f / 1}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 3, 0.0f / 1},
                                                             {2.0f / 3, 8.0f / 9, 2.0f / 3},
                                                             {2.0f / 1, 2.0f / 3, 0.0f / 1}},
                                                            {{3.0f / 1, 10.0f / 3, 2.0f / 1},
                                                             {8.0f / 3, 35.0f / 9, 16.0f / 3},
                                                             {3.0f / 1, 14.0f / 3, 5.0f / 1}}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_3x3_strided_uneven)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>(
            {{{{0.0f / 1, 1.0f / 2}, {2.0f / 3, 6.0f / 6}, {2.0f / 1, 0.0f / 2}},
              {{3.0f / 1, 7.0f / 2}, {8.0f / 3, 27.0f / 6}, {3.0f / 1, 11.0f / 2}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_3d)
{
    Shape shape_a{64, 3, 7, 8, 10};
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{5, 6, 4};
    Shape padding_above{6, 4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);

    auto cpu_f = make_shared<Function>(
        make_shared<op::AvgPool>(A, window_shape, move_strides, padding_below, padding_above, true),
        op::ParameterVector{A});
    auto int_f = make_shared<Function>(
        make_shared<op::AvgPool>(B, window_shape, move_strides, padding_below, padding_above, true),
        op::ParameterVector{B});
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}
