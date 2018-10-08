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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/log.hpp"
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

static const vector<element::Type> s_known_element_types = {element::from<float>(),
                                                            element::from<double>(),
                                                            element::from<int8_t>(),
                                                            element::from<int16_t>(),
                                                            element::from<int32_t>(),
                                                            element::from<int64_t>(),
                                                            element::from<uint8_t>(),
                                                            element::from<uint16_t>(),
                                                            element::from<uint32_t>(),
                                                            element::from<uint64_t>()};

class UnhandledOp : public ngraph::op::Op
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Op("Unsupported_op", {})
    {
        set_output_type(0, arg->get_element_type(), arg->get_shape());
    }
    shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
    {
        return make_shared<UnhandledOp>(new_args[0]);
    }
};

NGRAPH_TEST(${BACKEND_NAME}, unhandled_op)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor<float>(shape);
    ASSERT_THROW(backend->call_with_validate(f, {result}, {a}), unsupported_op);
}

NGRAPH_TEST(${BACKEND_NAME}, function_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B}, "funky func name");

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor<float>(shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, node_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    C->set_name("a node name");
    auto f = make_shared<Function>(C, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    auto D = A * B;
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out1 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out2 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out3 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out4 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out5 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out6 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out7 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    backend->call_with_validate(f, {out1, out2, out3, out4, out5, out6, out7}, {a, b});
    EXPECT_EQ(expectedC, read_vector<float>(out1));
    EXPECT_EQ(expectedC, read_vector<float>(out2));
    EXPECT_EQ(expectedD, read_vector<float>(out3));
    EXPECT_EQ(expectedD, read_vector<float>(out4));
    EXPECT_EQ(expectedC, read_vector<float>(out5));
    EXPECT_EQ(expectedE, read_vector<float>(out6));
    EXPECT_EQ(expectedE, read_vector<float>(out7));
}

NGRAPH_TEST(${BACKEND_NAME}, parameter_as_output)
{
    Shape shape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    vector<float> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> zero(shape_size(shape), 0);
    copy_data(a, expected);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, multiply)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Multiply>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A * B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, abc_int64)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto C = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::i64, shape);
    copy_data(c, vector<int64_t>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ((vector<int64_t>{50, 72, 98, 128}), read_vector<int64_t>(result));
}

// Multiple retrive values
NGRAPH_TEST(${BACKEND_NAME}, multiple_result)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto f =
        make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->create_tensor(element::f32, shape);
    auto r1 = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {r0, r1}, {a, b, c});

    EXPECT_EQ((vector<float>{6, 8, 10, 12}), read_vector<float>(r0));
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(r1));
}

NGRAPH_TEST(${BACKEND_NAME}, abs)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 0, 4.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, ceiling)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_overload)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A / B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto Y_out = f->get_output_op(0);
        auto Xs = f->get_parameters();
        auto C = std::make_shared<op::Parameter>(Y_out->get_element_type(), Y_out->get_shape());
        ngraph::autodiff::Adjoints adjoints(NodeVector{Y_out}, NodeVector{C});
        std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
        transform(
            Xs.begin(), Xs.end(), dYdXs.begin(), [C, &adjoints](const std::shared_ptr<Node>& X) {
                return adjoints.backprop_node(X);
            });
        std::vector<std::shared_ptr<op::Parameter>> params(Xs);
        params.push_back(C);

        auto bf = std::make_shared<Function>(dYdXs, params);

        return bf;
    };

    auto bf = make_external();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 2, 2, 2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{1, 1, 1, 1});

    auto resulta = backend->create_tensor(element::f32, shape);
    auto resultb = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(bf, {resulta, resultb}, {a, b, c});
    EXPECT_EQ((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta));
    EXPECT_EQ((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::i32, shape);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#pragma clang diagnostic ignored "-Wcovered-switch-default"

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_DEATH_IF_SUPPORTED(
        {
            try
            {
                backend->call_with_validate(f, {result}, {a, b});
            }
            catch (...)
            {
                abort();
            }
        },
        "");

#pragma clang diagnostic pop
}

NGRAPH_TEST(${BACKEND_NAME}, equal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greater)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greatereq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::GreaterEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, less)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Less>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_bool)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, log)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f});
    vector<float> loga{-2.07944154f,
                       -1.38629436f,
                       -0.69314718f,
                       0.00000000f,
                       0.69314718f,
                       1.38629436f,
                       2.07944154f,
                       2.77258872f};
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(loga, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto lrn = make_shared<op::LRN>(A, 1., 2., 1., 3);
    auto f = make_shared<Function>(lrn, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});

    vector<float> expected{0.f,
                           0.05325444f,
                           0.03402646f,
                           0.01869806f,
                           0.06805293f,
                           0.03287071f,
                           0.00509002f,
                           0.00356153f,
                           0.00174719f,
                           0.0012555f,
                           0.00322708f,
                           0.00235574f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6});
    auto result = backend->create_tensor(element::i32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 2, -8, 8, -5, 18448, 1, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 2, -8, 8, -5, 18448, 1, 280592}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6});
    auto result = backend->create_tensor(element::i32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 8, 4, 17, 0, 67635216, 2, 6}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 8, 4, 17, 0, 67635216, 2, 17179887632}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f, 8.75f, -8.75f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, notequal)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::NotEqual>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, select)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A - B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_2constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(NodeVector{A, A}, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result0 = backend->create_tensor(element::f32, shape);
    auto result1 = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result0, result1}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result0));
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto f = make_shared<Function>(make_shared<op::Abs>(A), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_multi_use)
{
    auto A = make_shared<op::Constant>(element::i32, Shape{}, std::vector<std::string>{"388"});
    auto f = make_shared<Function>(A, op::ParameterVector{});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::shared_ptr<runtime::Tensor> r1 = backend->create_tensor(element::i32, Shape{});
    backend->call_with_validate(f, {r1}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r1), std::vector<int>{388});

    std::shared_ptr<runtime::Tensor> r2 = backend->create_tensor(element::i32, Shape{});
    backend->call_with_validate(f, {r2}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r2), std::vector<int>{388});
}

NGRAPH_TEST(${BACKEND_NAME}, constant_broadcast)
{
    const string js =
        R"([{
       "name" : "Function_0",
       "ops" : [
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : [],
             "name" : "Parameter_4",
             "op" : "Parameter",
             "outputs" : ["Parameter_4"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : [],
             "name" : "Parameter_0",
             "op" : "Parameter",
             "outputs" : ["Parameter_0"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : [],
             "name" : "Constant_1",
             "op" : "Constant",
             "outputs" : ["Constant_1"],
             "shape" : [],
             "value" : ["0"]
           },
           {
             "axes" : [ 0, 1 ],
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : ["Constant_1"],
             "name" : "Broadcast_2",
             "op" : "Broadcast",
             "outputs" : ["Broadcast_2"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : [ "Parameter_0", "Broadcast_2" ],
             "name" : "Maximum_3",
             "op" : "Maximum",
             "outputs" : ["Maximum_3"]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false},
             "inputs" : [ "Maximum_3", "Parameter_4" ],
             "name" : "Multiply_5",
             "op" : "Multiply",
             "outputs" : ["Multiply_5"]
           }
       ],
       "parameters" : [ "Parameter_0", "Parameter_4" ],
       "result" : ["Multiply_5"],
       "result_shape" : [ 3, 4 ],
       "result_type" :
           {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true, "is_quantized" : false}
    }])";
    stringstream ss(js);

    shared_ptr<Function> f = ngraph::deserialize(ss);

    // max(x,broadcast(Constant(0)))
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // If this compiles it works
}

NGRAPH_TEST(${BACKEND_NAME}, function_call)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto g =
        make_shared<Function>(make_shared<op::FunctionCall>(f, NodeVector{X + Y, Y + Z, Z + X}) +
                                  make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}),
                              op::ParameterVector{X, Y, Z});

    // Now call g on some test vectors.
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(g, {result}, {x, y, z});
    EXPECT_EQ((vector<float>{254, 368, 502, 656}), read_vector<float>(result));

    backend->call_with_validate(g, {result}, {y, x, z});
    EXPECT_EQ((vector<float>{278, 400, 542, 704}), read_vector<float>(result));

    backend->call_with_validate(g, {result}, {x, z, y});
    EXPECT_EQ((vector<float>{194, 296, 418, 560}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::f32), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_uint16_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::u16, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::f32), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u16, shape);
    copy_data(a, vector<uint16_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_float32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::boolean),
                                   op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

// Trivial case with no reduction axes.
NGRAPH_TEST(${BACKEND_NAME}, reduce_trivial)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, {});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_to_scalar)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_columns)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};

    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{66});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{66, 66, 66}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{66}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_cols_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{77});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{77, 77}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{77}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_vector_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{88});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{88}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{88}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_to_scalar_zero_by_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{99});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{99}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{99}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_3d_to_vector)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x*y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(make_shared<op::Multiply>(f_A, f_B), op::ParameterVector{f_A, f_B});

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(A, B, f, AxisSet{0, 1}),
                                   op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sin)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{0.00000000f,
                                                0.24740396f,
                                                -0.24740396f,
                                                0.47942554f,
                                                -0.47942554f,
                                                0.84147098f,
                                                -0.84147098f,
                                                0.90929743f,
                                                -0.90929743f,
                                                -0.75680250f,
                                                0.75680250f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cos)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1.00000000f,
                                                0.96891242f,
                                                0.96891242f,
                                                0.87758256f,
                                                0.87758256f,
                                                0.54030231f,
                                                0.54030231f,
                                                -0.41614684f,
                                                -0.41614684f,
                                                -0.65364362f,
                                                -0.65364362f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, tan)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{0.00000000f,
                                                0.25534192f,
                                                -0.25534192f,
                                                0.54630249f,
                                                -0.54630249f,
                                                1.55740772f,
                                                -1.55740772f,
                                                -2.18503986f,
                                                2.18503986f,
                                                1.15782128f,
                                                -1.15782128f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, asin)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{-1.57079633f,
                                                -0.84806208f,
                                                -0.52359878f,
                                                -0.25268026f,
                                                -0.12532783f,
                                                0.00000000f,
                                                0.12532783f,
                                                0.25268026f,
                                                0.52359878f,
                                                0.84806208f,
                                                1.57079633f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, acos)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{3.14159265f,
                                                2.41885841f,
                                                2.09439510f,
                                                1.82347658f,
                                                1.69612416f,
                                                1.57079633f,
                                                1.44546850f,
                                                1.31811607f,
                                                1.04719755f,
                                                0.72273425f,
                                                0.00000000f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, atan)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{-1.32581766f,
                                                -1.10714872f,
                                                -0.78539816f,
                                                -0.46364761f,
                                                -0.24497866f,
                                                0.00000000f,
                                                0.24497866f,
                                                0.46364761f,
                                                0.78539816f,
                                                1.10714872f,
                                                1.32581766f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, sinh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cosh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return coshf(x); });

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, tanh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanhf(x); });

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, exp)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-4, -3, -2, -1, 0, 1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)},
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto r = op::Constant::create(element::f32, Shape{}, {4.75});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ(vector<float>{4.75f}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto r = op::Constant::create(element::i64, Shape{}, {2112});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, Shape{});

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ(vector<int64_t>{2112}, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::f32, shape, {4.75, 4.5, -5.25, 0.0});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<float>{4.75f, 4.5f, -5.25f, 0.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::i64, shape, {2112, 1848, 1776, 1964});
    auto f = make_shared<Function>(r, op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<int64_t>{2112, 1848, 1776, 1964}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_TRUE(test::all_close(vector<float>{1, 1, 729, 125}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_equality_bool)
{
    Shape shape{4};
    // auto A = make_shared<op::Parameter>(element::boolean, shape);
    // auto B = make_shared<op::Parameter>(element::boolean, shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<char>{true, false, true, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<float>{4, 2, 9, 10, 100, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{808});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{808}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_matrix_inplace)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto abs_A = make_shared<op::Abs>(A);

    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(abs_A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto abs_r = make_shared<op::Abs>(r);
    auto f = make_shared<Function>(abs_r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{12};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(
        (vector<float>{0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 14, 15}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 1, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_fp_nonint_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.1f});
    auto result = backend->create_tensor(element::f32, shape_r);

    try
    {
        backend->call_with_validate(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_oob_in_3)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3};
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{3000000});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call_with_validate(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 8};
    auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(
        (vector<int32_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call_with_validate(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_far_oob)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3000000, 2, 1, 0});
    auto result = backend->create_tensor(element::i32, shape_r);

    try
    {
        backend->call_with_validate(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: value is out of category range"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 3, 3};
    auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->create_tensor(element::i32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_fp)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(
        (vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_fp_nonint)
{
    Shape shape_a{8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{8, 3};
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1.01f, 0});
    auto result = backend->create_tensor(element::f32, shape_r);

    try
    {
        backend->call_with_validate(f, {result}, {a});
    }
    catch (const std::exception& e)
    {
        EXPECT_EQ(e.what(), std::string("One-hot: non-integral value in input"));
    }
    catch (...)
    {
        FAIL() << "Expected a std::out_of_range exception";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{921, 922, 925, 926, 937, 938, 941, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

                             16, 17, 18, 19, 20, 921, 922, 23, 24, 925, 926, 27, 28, 29, 30, 31,

                             32, 33, 34, 35, 36, 937, 938, 39, 40, 941, 942, 43, 44, 45, 46, 47,

                             48, 49, 50, 51, 52, 53,  54,  55, 56, 57,  58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 902, 908, 910, 932, 934, 940, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

                             16,  17, 18,  19, 20, 21, 22, 23, 24,  25, 26,  27, 28, 29, 30, 31,

                             932, 33, 934, 35, 36, 37, 38, 39, 940, 41, 942, 43, 44, 45, 46, 47,

                             48,  49, 50,  51, 52, 53, 54, 55, 56,  57, 58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 903, 908, 911, 932, 935, 940, 943});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, not)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 2, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), op::ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    backend->call_with_validate(f, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

template <typename OP>
void make_unary_empty_test(const string& backend_name)
{
    Shape shape{0};

    op::ParameterVector params;
    NodeVector result_list;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(s_known_element_types[i], shape);
        params.push_back(p);
        result_list.push_back(make_shared<OP>(p));
    }

    auto f = make_shared<Function>(result_list, params);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
    }

    backend->call_with_validate(f, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
}

template <typename OP>
void make_binary_empty_test(const string& backend_name, bool is_comparison = false)
{
    Shape shape{0};
    op::ParameterVector A;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        A.push_back(make_shared<op::Parameter>(s_known_element_types[i], shape));
    }

    NodeVector result_list;
    for (shared_ptr<op::Parameter> p : A)
    {
        result_list.push_back(make_shared<OP>(p, p));
    }

    auto f = make_shared<Function>(result_list, A);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        if (is_comparison)
        {
            outputs.push_back(backend->create_tensor(element::from<char>(), shape));
        }
        else
        {
            outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        }
    }

    backend->call_with_validate(f, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    if (is_comparison)
    {
        EXPECT_EQ(read_vector<char>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[9]).size(), 0);
    }
    else
    {
        EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_abs)
{
    make_unary_empty_test<op::Abs>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_ceiling)
{
    make_unary_empty_test<op::Ceiling>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_exp)
{
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_floor)
{
    make_unary_empty_test<op::Floor>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_log)
{
    make_unary_empty_test<op::Log>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_negative)
{
    make_unary_empty_test<op::Negative>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not)
{
    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::from<char>(), shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::from<char>(), shape);
    auto result = backend->create_tensor(element::from<char>(), shape);

    backend->call_with_validate(f, {result}, {a});

    auto in_vec = read_vector<char>(a);
    auto out_vec = read_vector<char>(result);

    EXPECT_EQ(in_vec.size(), 0);
    EXPECT_EQ(out_vec.size(), 0);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sign)
{
    make_unary_empty_test<op::Sign>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sqrt)
{
    make_unary_empty_test<op::Sqrt>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sin)
{
    make_unary_empty_test<op::Sin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sinh)
{
    make_unary_empty_test<op::Sinh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cos)
{
    make_unary_empty_test<op::Cos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cosh)
{
    make_unary_empty_test<op::Cosh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tan)
{
    make_unary_empty_test<op::Tan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tanh)
{
    make_unary_empty_test<op::Tanh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_asin)
{
    make_unary_empty_test<op::Asin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_acos)
{
    make_unary_empty_test<op::Acos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_atan)
{
    make_unary_empty_test<op::Atan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_add)
{
    make_binary_empty_test<op::Add>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_divide)
{
    make_binary_empty_test<op::Divide>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_eq)
{
    make_binary_empty_test<op::Equal>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greater)
{
    make_binary_empty_test<op::Greater>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greatereq)
{
    make_binary_empty_test<op::GreaterEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_less)
{
    make_binary_empty_test<op::Less>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_lesseq)
{
    make_binary_empty_test<op::LessEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_maximum)
{
    make_binary_empty_test<op::Maximum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_minimum)
{
    make_binary_empty_test<op::Minimum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_multiply)
{
    make_binary_empty_test<op::Multiply>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not_equal)
{
    make_binary_empty_test<op::NotEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_power)
{
    make_binary_empty_test<op::Power>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_subtract)
{
    make_binary_empty_test<op::Subtract>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_outlining)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto conv2 = make_shared<op::Convolution>(conv1,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto f = make_shared<Function>(conv2, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(vector<float>{expected_result}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, computation_reuse)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv = make_shared<op::Convolution>(A,
                                             B,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool = make_shared<op::AvgPool>(conv, pool_shape);
    auto bias = make_shared<op::Broadcast>(
        op::Constant::create(element::f32, Shape{}, {2.14}), shape_r, AxisSet{0, 1, 2, 3});
    auto result_op = make_shared<op::Result>(pool + bias);
    auto f = make_shared<Function>(ResultVector{result_op}, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> input(64, 1.0f);
    vector<float> weights(512, 0.5f);
    vector<float> rv(128);

    auto a = backend->create_tensor(element::f32, shape_a, input.data());
    auto b = backend->create_tensor(element::f32, shape_b, weights.data());
    auto result = backend->create_tensor(element::f32, shape_r, rv.data());

    backend->call_with_validate(f, {result}, {a, b});

    vector<float> rv_saved(rv);

    b->set_stale(false);
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(rv_saved, rv);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h2w2)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    backend->call_with_validate(func, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    backend->call_with_validate(func, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_bprop_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::SigmoidBackprop>(input, delta);
    auto func = make_shared<Function>(sigmoid_node, op::ParameterVector{input, delta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, delta->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{1.0f, 1.0f, 1.0f, 1.0f};

    copy_data(a, dataA);
    copy_data(b, dataB);
    backend->call_with_validate(func, {result}, {a, b});

    vector<float> expected{0.196612f, 0.0176627f, 0.196612f, 0.0176627f};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, op::ParameterVector{B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    backend->call_with_validate(f, {result}, {b});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dbackprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};

    backend->call_with_validate(f, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dbackprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, op::ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    backend->call_with_validate(f, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0, 1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-3, -2, -1, 0, 1, 2});
    auto result = backend->create_tensor(element::f32, shape);

    auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{
        expf(-3) / d, expf(-2) / d, expf(-1) / d, expf(0) / d, expf(1) / d, expf(2) / d};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));

    // empty AxisSet is the same as "full" AxisSet
    f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{}), op::ParameterVector{A});
    backend = runtime::Backend::create("${BACKEND_NAME}");

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-1);
    auto d1 = expf(-20) + expf(-2);
    auto d2 = expf(-30) + expf(-3);
    auto d3 = expf(-40) + expf(-4);
    auto d4 = expf(-50) + expf(-5);
    auto d5 = expf(-60) + expf(-6);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d3,
                           expf(-50) / d4,
                           expf(-60) / d5,
                           expf(-1) / d0,
                           expf(-2) / d1,
                           expf(-3) / d2,
                           expf(-4) / d3,
                           expf(-5) / d4,
                           expf(-6) / d5};

    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{1}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-20) + expf(-30);
    auto d1 = expf(-40) + expf(-50) + expf(-60);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d0,
                           expf(-30) / d0,
                           expf(-40) / d1,
                           expf(-50) / d1,
                           expf(-60) / d1};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_2)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-40);
    auto d1 = expf(-20) + expf(-50);
    auto d2 = expf(-30) + expf(-60);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d0,
                           expf(-50) / d1,
                           expf(-60) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d_trivial)
{
    Shape shape{1, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{1, 1, 1, 1, 1, 1};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_underflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto low = std::numeric_limits<float>::lowest();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{low, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(low) + expf(3);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    backend->call_with_validate(f, {result}, {a});
    vector<float> expected{
        expf(low) / d0, expf(1) / d1, expf(2) / d2, expf(3) / d0, expf(4) / d1, expf(5) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, multiple_backends)
{
    Shape shape{2, 2};
    auto A1 = make_shared<op::Parameter>(element::f32, shape);
    auto B1 = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A1 + B1, op::ParameterVector{A1, B1});

    auto A2 = make_shared<op::Parameter>(element::f32, shape);
    auto B2 = make_shared<op::Parameter>(element::f32, shape);
    auto g = make_shared<Function>(A2 * B2, op::ParameterVector{A2, B2});

    auto backend1 = runtime::Backend::create("${BACKEND_NAME}");

    auto backend2 = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result1 = backend1->create_tensor(element::f32, shape);

    shared_ptr<runtime::Tensor> a2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result2 = backend2->create_tensor(element::f32, shape);

    copy_data(a1, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b1, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    copy_data(a2, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b2, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend1->call_with_validate(f, {result1}, {a1, b1});
    EXPECT_EQ(read_vector<float>(result1),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());

    backend2->call_with_validate(g, {result2}, {a2, b2});
    EXPECT_EQ(read_vector<float>(result2),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, tensorview_custom_mem)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        return f;
    };

    auto f = make_external();

    vector<float> av{2, 4, 8, 16};
    vector<float> bv{1, 2, 4, 8};
    // use custom mem with tensorview, no need to copy data
    auto a = backend->create_tensor(element::f32, shape, av.data());
    auto b = backend->create_tensor(element::f32, shape, bv.data());

    // use custom mem with result tensorview
    vector<float> rv{0, 0, 0, 0};
    auto result = backend->create_tensor(element::f32, shape, rv.data());

    // result should be in memory without needing explict read
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), rv);
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);
    auto d = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(backend->call_with_validate(f, {a}, {c, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_and)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::And>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 0, 0, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_or)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Or>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 1, 1, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n2c3h4w2)
{
    Shape shape{2, 3, 4, 2};
    Shape seq_len_shape{4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 2;
    size_t sequence_axis = 1;
    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{
        0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0, 7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
    };

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    std::vector<int> expected{
        0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0, 7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,

        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0};

    copy_data(a, input);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4c3h2w2)
{
    Shape shape{4, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 1;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> seq_lenghts{1, 2, 3, 3};
    copy_data(b, seq_lenghts);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                              12, 13, 14, 15, 20, 21, 22, 23, 32, 33, 34, 35, 28, 29, 30, 31,
                              24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39};

    copy_data(a, input);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4d2c3h2w2)
{
    Shape shape{4, 2, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 2;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27,
                              32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                              64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                              80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95};

    copy_data(a, input);

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

// Trivial case.
NGRAPH_TEST(${BACKEND_NAME}, argmin_trivial)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::ArgMin>(A, 0, element::i32), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int>{3, 2, 1}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_trivial)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::ArgMax>(A, 0, element::i32), op::ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int>{1, 3, 0}), read_vector<int>(result));
}
