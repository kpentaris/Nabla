// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_


#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"


namespace nbl
{
namespace hlsl
{
#ifndef __HLSL_VERSION // CPP
#define ALIAS_STD(NAME,OP) template<typename T> struct NAME : std::NAME<T> {
#else
#define ALIAS_STD(NAME,OP) template<typename T> struct NAME { \
    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) \
    { \
        return lhs OP rhs; \
    }
#endif


ALIAS_STD(bit_and,&)

    NBL_CONSTEXPR_STATIC_INLINE T identity = bit_cast<scalar_type<T>::type>(~0ull); // TODO: need a `all_components<T>` (not in cpp_compat) which can create vectors and matrices with all members set to scalar
};

ALIAS_STD(bit_or,|)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(bit_xor,^)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(plus,+)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};


ALIAS_STD(multiplies,*)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(1);
};


ALIAS_STD(greater,>) };
ALIAS_STD(less,<) };
ALIAS_STD(greater_equal,>=) };
ALIAS_STD(less_equal,<=) };

#undef ALIAS_STD


// Min and Max don't use ALIAS_STD because they don't exist in STD
// TODO: implement as mix(rhs<lhs,lhs,rhs) (SPIR-V intrinsic from the extended set & glm on C++)
template<typename T>
struct minimum
{
    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
    {
        return rhs<lhs ? rhs:lhs;
    }

    NBL_CONSTEXPR_STATIC_INLINE T identity = numeric_limits<scalar_type<T>::type>::max; // TODO: `all_components<T>`
};

template<typename T>
struct maximum
{
    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
    {
        return lhs<rhs ? rhs:lhs;
    }

    NBL_CONSTEXPR_STATIC_INLINE T identity = numeric_limits<scalar_type<T>::type>::lowest; // TODO: `all_components<T>`
};

}
}

#endif