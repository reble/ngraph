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

#include <iostream>

/// \brief This file is for static initialization of the ngraph shared library

namespace ngraph
{
    class StaticInitializer;
}

class ngraph::StaticInitializer
{
public:
    StaticInitializer() { std::cout << "*************************** ngraph ctor\n"; }
    ~StaticInitializer() { std::cout << "*************************** ngraph dtor\n"; }
};

static ngraph::StaticInitializer s_ngraph_static_initializer;
