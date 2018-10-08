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

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 0, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{6, 5, 4, 3, 2, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 3, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{6, 5, 4}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 0, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 3, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 0, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 9, 4, 8, 2, 11, 7, 6, 3, 5, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 2, 0, 1}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 9, 4, 11, 7, 6, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 1, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 2, 2}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 11, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 0, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 0, 1, 1, 0, 0, 1, 2, 2}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 10, 4, 12, 9, 5, 1, 6, 3, 11, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 1, 0, 0, 1}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 10, 4, 5, 1, 6, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 1, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 5, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 4, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3, 2, 0, 2, 3, 2, 1}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10, 9, 8, 7, 6, 2, 5, 3, 1, 4}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 2, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10, 9, 8, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, true);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 4, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2, 1, 1, 3, 0, 3, 0}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4, 6, 2, 5, 9, 8, 7, 12, 11, 10}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 2, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4, 6, 2, 5}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, false);
    auto f0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), op::ParameterVector{A});
    auto f1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    backend->call_with_validate(f0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1}), read_vector<int32_t>(result0));
    backend->call_with_validate(f1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4}), read_vector<float>(result1));
}
