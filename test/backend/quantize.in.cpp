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

NGRAPH_TEST(${BACKEND_NAME}, quantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  1  1  2  2  3  3  4  4  5  5   6
    // plus offset                    1  1  1  1  1  1  1  1  1  1  1   1
    // equals                         1  2  2  3  3  4  4  5  5  6  6   7

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7});
    // minus offset                   1  1  1  1  1  1  1  1  1  1  1  1
    // eqauls                         0  1  1  2  2  3  3  4  4  5  5  6
    // multiplied by scale            2  2  2  2  2  2  2  2  2  2  2  2
    // equals                         0  2  2  4  4  6  6  8  8 10 10 12

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {10, 20, 30, 40});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divided by scale               2  2  2  3  3  3  4  4  4  5  5   5
    // equals (rounded)               0  1  1  1  1  2  2  2  2  2  2   2
    // plus offset                   10 10 10 20 20 20 30 30 30 40 40  40
    // equals                        10 11 11 21 21 22 32 32 32 42 42  42

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {10, 20, 30, 40});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42});
    // minus offset                   10  10  10  20  20  20  30  30  30  40  40  40
    // equals                          0   1   1   1   1   2   2   2   2   2   2   2
    // multiplied by scale             2   2   2   3   3   3   4   4   4   5   5   5
    // equals                          0   2   2   3   3   6   8   8   8  10  10  10

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 2, 2, 3, 3, 6, 8, 8, 8, 10, 10, 10}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0  -1  1  -2  2  -3  3  -4  4  -5  5   -6
    // plus offset                    1   1  1   1  1   1  1   1  1   1  1    1
    // equals                         1   0  2  -1  3  -2  4  -3  5  -4  6   -5

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 0, 2, -1, 3, -2, 4, -3, 5, -4, 6, -5}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i8;
    auto output_type = element::f32;

    typedef int8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 0, 2, -1, 3, -2, 4, -3, 5, -4, 6, -5});
    // minus offset                   1  1  1   1  1   1  1   1  1   1  1   1
    // equals                         0 -1  1  -2  2  -3  3  -4  4  -5  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0 -2  2  -4  4  -6  6  -8  8 -10 10 -12

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, -2, 2, -4, 4, -6, 6, -8, 8, -10, 10, -12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {0.00001});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, op::ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});

    backend->call_with_validate(f, {y}, {x});
    EXPECT_EQ(
        (vector<output_c_type>{1, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128}),
        read_vector<output_c_type>(y));
}
