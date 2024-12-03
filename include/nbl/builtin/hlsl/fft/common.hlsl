#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/complex.hlsl>

#ifndef __HLSL_VERSION
#include <nbl/core/math/intutil.h>

namespace nbl
{
namespace hlsl
{
namespace fft
{

static inline uint32_t3 padDimensions(uint32_t3 dimensions, std::span<uint16_t> axes, bool realFFT = false)
{
    uint16_t axisCount = 0;
    for (auto i : axes)
    {
        dimensions[i] = core::roundUpToPoT(dimensions[i]);
        if (realFFT && !axisCount++)
            dimensions[i] /= 2;
    }
    return dimensions;
}

static inline uint64_t getOutputBufferSize(const uint32_t3& inputDimensions, uint32_t numChannels, std::span<uint16_t> axes, bool realFFT = false, bool halfFloats = false)
{
    auto paddedDims = padDimensions(inputDimensions, axes);
    uint64_t numberOfComplexElements = paddedDims[0] * paddedDims[1] * paddedDims[2] * numChannels;
    return numberOfComplexElements * (halfFloats ? sizeof(complex_t<float16_t>) : sizeof(complex_t<float32_t>));
}


}
}
}

#else

#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace fft 
{

// Computes the kth element in the group of N roots of unity
// Notice 0 <= k < N/2, rotating counterclockwise in the forward (DIF) transform and clockwise in the inverse (DIT)
template<bool inverse, typename Scalar>
complex_t<Scalar> twiddle(uint32_t k, uint32_t halfN)
{
    complex_t<Scalar> retVal;
    const Scalar kthRootAngleRadians = numbers::pi<Scalar> * Scalar(k) / Scalar(halfN);
    retVal.real( cos(kthRootAngleRadians) );
    if (! inverse)
        retVal.imag( sin(-kthRootAngleRadians) );
    else
        retVal.imag( sin(kthRootAngleRadians) );
    return retVal;                         
}

template<bool inverse, typename Scalar> 
struct DIX 
{ 
    static void radix2(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
    {
        plus_assign< complex_t<Scalar> > plusAss;
        //Decimation in time - inverse           
        if (inverse) {
            complex_t<Scalar> wHi = twiddle * hi;
            hi = lo - wHi;
            plusAss(lo, wHi);            
        }
        //Decimation in frequency - forward   
        else {
            complex_t<Scalar> diff = lo - hi;
            plusAss(lo, hi);
            hi = twiddle * diff; 
        }
    }                                              
};

template<typename Scalar>
using DIT = DIX<true, Scalar>;

template<typename Scalar>
using DIF = DIX<false, Scalar>;

// ------------------------------------------------- Utils ---------------------------------------------------------
// 
// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
template<typename Scalar>
void unpack(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
{
    complex_t<Scalar> x = (lo + conj(hi)) * Scalar(0.5);
    hi = rotateRight<Scalar>(lo - conj(hi)) * Scalar(0.5);
    lo = x;
}

// Bit-reverses T as a binary string of length given by Bits
template<typename T, uint16_t Bits NBL_FUNC_REQUIRES(is_integral_v<T> && Bits <= sizeof(T) * 8)
T bitReverse(T value)
{
    return glsl::bitfieldReverse<uint32_t>(value) >> (sizeof(T) * 8 - Bits);
}

}
}
}

#endif

#endif