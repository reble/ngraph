/*******************************************************************************
* Copyright 2018 Intel Corporation
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
*******************************************************************************/

#include <numeric>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/cpu/op/quantized_conv_relu.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedConvolutionRelu::QuantizedConvolutionRelu(
    const std::shared_ptr<op::QuantizedConvolution>& qconv)
    : Op("QuantizedConvolutionRelu",
         check_single_output_args({qconv->get_argument(0),
                                   qconv->get_argument(1),
                                   qconv->get_argument(2),
                                   qconv->get_argument(3)}))
    , m_window_movement_strides(qconv->get_window_movement_strides())
    , m_window_dilation_strides(qconv->get_window_dilation_strides())
    , m_padding_below(qconv->get_padding_below())
    , m_padding_above(qconv->get_padding_above())
    , m_data_dilation_strides(qconv->get_data_dilation_strides())
{
    constructor_validate_and_infer_types();

    this->m_scale = qconv->get_scale();
    this->m_offset = qconv->get_offset();

    set_output_size(3);
    set_output_type(0, element::u8, qconv->get_shape());
    set_output_type(1, element::f32, Shape{1});
    set_output_type(2, element::f32, Shape{1});
}

op::QuantizedConvolutionRelu::QuantizedConvolutionRelu(const std::shared_ptr<Node>& data_batch,
                                                       const std::shared_ptr<Node>& filters,
                                                       const Strides& window_movement_strides,
                                                       const Strides& window_dilation_strides,
                                                       const CoordinateDiff& padding_below,
                                                       const CoordinateDiff& padding_above,
                                                       const Strides& data_dilation_strides,
                                                       const std::shared_ptr<Node> scale,
                                                       const std::shared_ptr<Node> offset)
    : Op("QuantizedConvolutionRelu", check_single_output_args({data_batch, filters, scale, offset}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch->get_shape();
    auto& filters_shape = filters->get_shape();

    auto scale_const_op = std::static_pointer_cast<ngraph::op::Constant>(scale);
    auto offset_const_op = std::static_pointer_cast<ngraph::op::Constant>(offset);
    float scale_val = *(static_cast<float const*>(scale_const_op->get_data_ptr()));
    float offset_val = *(static_cast<float const*>(offset_const_op->get_data_ptr()));
    this->m_scale = scale_val;
    this->m_offset = offset_val;

    set_output_size(3);
    set_output_type(0,
                    element::u8,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         0, /* batch_axis_data,              */
                                                         1, /* input_channel_axis_data,      */
                                                         1, /* input_channel_axis_filters,   */
                                                         0, /* output_channel_axis_filters,  */
                                                         0, /* batch_axis_result,            */
                                                         1  /* output_channel_axis_result,   */
                                                         ));

    set_output_type(1, element::f32, Shape{1});
    set_output_type(2, element::f32, Shape{1});
}

std::shared_ptr<Node>
    op::QuantizedConvolutionRelu::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return std::shared_ptr<Node>(new QuantizedConvolutionRelu(new_args.at(0),
                                                              new_args.at(1),
                                                              get_window_movement_strides(),
                                                              get_window_dilation_strides(),
                                                              get_padding_below(),
                                                              get_padding_above(),
                                                              get_data_dilation_strides(),
                                                              new_args.at(2),
                                                              new_args.at(3)));
}
