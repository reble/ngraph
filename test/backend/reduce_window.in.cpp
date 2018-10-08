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
// The unit tests for ReduceWindow follow exactly what we test for MaxPool---but they use ReduceWindow to do it.
//
NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_1image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 5, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 4, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

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
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_1channel_1image_strided)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 8, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 3, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

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
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}
