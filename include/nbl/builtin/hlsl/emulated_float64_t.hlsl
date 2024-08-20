#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_

#include <nbl/builtin/hlsl/impl/emulated_float64_t_impl.hlsl>

namespace nbl
{
namespace hlsl
{
    template<bool FastMath = true, bool FlushDenormToZero = true>
    struct emulated_float64_t
    {
        using storage_t = uint64_t;

        storage_t data;

        // constructors
        /*static emulated_float64_t create(uint16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(emulated_float64_t<FastMath, FlushDenormToZero> val)
        {
            return createPreserveBitPattern(val.data);
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(int32_t val)
        {
            return emulated_float64_t<FastMath, FlushDenormToZero>::createPreserveBitPattern(impl::castToUint64WithFloat64BitPattern(int64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(int64_t val)
        {
            return emulated_float64_t<FastMath, FlushDenormToZero>::createPreserveBitPattern(impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(uint32_t val)
        {
            return emulated_float64_t<FastMath, FlushDenormToZero>::createPreserveBitPattern(impl::castToUint64WithFloat64BitPattern(uint64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(uint64_t val)
        {
            return emulated_float64_t<FastMath, FlushDenormToZero>::createPreserveBitPattern(impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(float32_t val)
        {
            emulated_float64_t<FastMath, FlushDenormToZero> output;
            output.data = impl::castToUint64WithFloat64BitPattern(val);
            return output;
        }

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t<FastMath, FlushDenormToZero> create(float64_t val)
        {
            return createPreserveBitPattern(bit_cast<uint64_t>(float64_t(val)));
#ifdef __HLSL_VERSION
            emulated_float64_t retval;
            uint32_t lo, hi;
            asuint(val, lo, hi);
            retval.data = (uint64_t(hi) << 32) | lo;
            return retval;
#else
            return createPreserveBitPattern(reinterpret_cast<uint64_t&>(val));
#endif
        }
        
        // TODO: unresolved external symbol imath_half_to_float_table
        /*static emulated_float64_t create(float16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        NBL_CONSTEXPR_STATIC_INLINE emulated_float64_t createPreserveBitPattern(uint64_t val)
        {
            return emulated_float64_t(val);
        }

        inline float getAsFloat32()
        {
            int exponent = ieee754::extractExponent(data);
            if (!FastMath)
            {
                if (exponent > 127)
                    return bit_cast<float>(ieee754::traits<float>::inf);
                if (exponent < -126)
                    return -bit_cast<float>(ieee754::traits<float>::inf);
                if (tgmath::isnan(data))
                    return bit_cast<float>(ieee754::traits<float>::quietNaN);
            }

            //return float(bit_cast<double>(data));
            // TODO: fix
            uint32_t sign = uint32_t((data & ieee754::traits<float64_t>::signMask) >> 32);
            uint32_t biasedExponent = uint32_t(exponent + ieee754::traits<float>::exponentBias) << ieee754::traits<float>::mantissaBitCnt;
            uint32_t mantissa = uint32_t(data >> (ieee754::traits<float64_t>::mantissaBitCnt - ieee754::traits<float>::mantissaBitCnt)) & ieee754::traits<float>::mantissaMask;

            return bit_cast<float>(sign | biasedExponent | mantissa);

        }

#if 0
        uint64_t shiftLeftAllowNegBitCnt(uint64_t val, int n)
        {
            if (n < 0)
                return val >> -n;
            else
                return val << n;
        }
#endif

        // arithmetic operators
        emulated_float64_t operator+(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
#if 0
            {
                uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
                uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
                uint64_t lhsMantissa = ieee754::extractMantissa(data);
                uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                if (tgmath::isnan(data) || tgmath::isnan(rhs.data))
                    return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN);
                /*if (std::isinf(lhs) || std::isinf(rhs))
                {
                    if (std::isinf(lhs) && !std::isinf(rhs))
                        return lhs;
                    if (std::isinf(rhs) && !std::isinf(lhs))
                        return rhs;
                    if (rhs == lhs)
                        return rhs;

                    return nan();
                }*/

                int rp = min(ieee754::extractExponent(data), ieee754::extractExponent(rhs.data)) - ieee754::traits<float64_t>::mantissaBitCnt;

                uint64_t lhsRealMantissa = lhsMantissa | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);
                uint64_t rhsRealMantissa = rhsMantissa | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);
                uint64_t lhsSignTmp = lhsSign >> (52 + 11);
                uint64_t rhsSignTmp = rhsSign >> (52 + 11);

                uint64_t sign = 0u;
                if (lhsSign != rhsSign)
                {
                    uint64_t _min = max(data, rhs.data);
                    uint64_t _max = min(data, rhs.data);
                    uint64_t minAbs = _min ^ ieee754::traits<float64_t>::signMask;
                    if (minAbs > _max)
                        sign = ieee754::traits<float64_t>::signMask;

                }

                int64_t lhsMantissaTmp = (shiftLeftAllowNegBitCnt(lhsRealMantissa, lhsBiasedExp - rp - ieee754::traits<float64_t>::mantissaBitCnt - ieee754::traits<float64_t>::exponentBias) ^ (-lhsSignTmp)) + lhsSignTmp;
                int64_t rhsMantissaTmp = (shiftLeftAllowNegBitCnt(rhsRealMantissa, rhsBiasedExp - rp - ieee754::traits<float64_t>::mantissaBitCnt - ieee754::traits<float64_t>::exponentBias) ^ (-rhsSignTmp)) + rhsSignTmp;

                uint64_t addTmp = bit_cast<uint64_t>(lhsMantissaTmp + rhsMantissaTmp);

                // renormalize
                if (!FastMath && false) // TODO: hande nan
                {

                }
                else
                {
#ifndef __HLSL_VERSION
                    int l2 = log2(double(addTmp));
#else
                    int intl2 = 0;
#endif

                    if (!FastMath && (rp + l2 + 1 < nbl::hlsl::numeric_limits<float64_t>::min_exponent))
                    {
                        return createPreserveBitPattern(impl::assembleFloat64(0, ieee754::traits<float64_t>::exponentMask, 0));
                    }
                    else
                    {
                        rp = addTmp ? l2 + rp + ieee754::traits<float64_t>::exponentBias : 0;
                        return createPreserveBitPattern(impl::assembleFloat64(
                            sign,
                            (uint64_t(rp) << ieee754::traits<float64_t>::mantissaBitCnt) & ieee754::traits<float64_t>::exponentMask,
                            shiftLeftAllowNegBitCnt(addTmp, (ieee754::traits<float64_t>::mantissaBitCnt - l2)) & ieee754::traits<float64_t>::mantissaMask)
                        );
                    }
                }
            }
#endif

            if (FlushDenormToZero)
            {
                emulated_float64_t retval = createPreserveBitPattern(0u);

                uint64_t mantissa;
                uint32_t3 mantissaExtended;
                int biasedExp;

                uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
                uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
                uint64_t lhsMantissa = ieee754::extractMantissa(data);
                uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                int expDiff = lhsBiasedExp - rhsBiasedExp;

                if (lhsSign == rhsSign)
                {
                    if (expDiff == 0)
                    {
                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            bool propagate = (lhsMantissa | rhsMantissa) != 0u;
                            return createPreserveBitPattern(tgmath::lerp(data, impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        mantissa = lhsMantissa + rhsMantissa;
                        if (lhsBiasedExp == 0)
                            return createPreserveBitPattern(impl::assembleFloat64(lhsSign, 0, mantissa));
                        mantissaExtended.xy = impl::packUint64(mantissa);
                        mantissaExtended.x |= 0x00200000u;
                        mantissaExtended.z = 0u;
                        biasedExp = lhsBiasedExp;

                        mantissaExtended = impl::shift64ExtraRightJamming(mantissaExtended, 1);
                    }
                    else
                    {
                        if (expDiff < 0)
                        {
                            swap<uint64_t>(lhsMantissa, rhsMantissa);
                            swap<int>(lhsBiasedExp, rhsBiasedExp);
                        }

                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            const bool propagate = (lhsMantissa) != 0u;
                            return createPreserveBitPattern(tgmath::lerp(ieee754::traits<float64_t>::exponentMask | lhsSign, impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        expDiff = tgmath::lerp(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                        rhsMantissa = tgmath::lerp(rhsMantissa | (1ull << 52), rhsMantissa, rhsBiasedExp == 0);
                        const uint32_t3 shifted = impl::shift64ExtraRightJamming(uint32_t3(impl::packUint64(rhsMantissa), 0u), expDiff);
                        rhsMantissa = impl::unpackUint64(shifted.xy);
                        mantissaExtended.z = shifted.z;
                        biasedExp = lhsBiasedExp;

                        lhsMantissa |= (1ull << 52);
                        mantissaExtended.xy = impl::packUint64(lhsMantissa + rhsMantissa);
                        --biasedExp;
                        if (!(mantissaExtended.x < 0x00200000u))
                        {
                            mantissaExtended = impl::shift64ExtraRightJamming(mantissaExtended, 1);
                            ++biasedExp;
                        }

                        return createPreserveBitPattern(impl::roundAndPackFloat64(lhsSign, biasedExp, mantissaExtended.xyz));
                    }

                    // cannot happen but compiler cries about not every path returning value
                    return createPreserveBitPattern(impl::roundAndPackFloat64(lhsSign, biasedExp, mantissaExtended));
                }
                else
                {
                    lhsMantissa = impl::shortShift64Left(lhsMantissa, 10);
                    rhsMantissa = impl::shortShift64Left(rhsMantissa, 10);

                    if (expDiff != 0)
                    {
                        uint32_t2 frac;

                        if (expDiff < 0)
                        {
                            swap<uint64_t>(lhsMantissa, rhsMantissa);
                            swap<int>(lhsBiasedExp, rhsBiasedExp);
                            lhsSign ^= ieee754::traits<float64_t>::signMask;
                        }

                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            bool propagate = lhsMantissa != 0u;
                            return createPreserveBitPattern(tgmath::lerp(impl::assembleFloat64(lhsSign, ieee754::traits<float64_t>::exponentMask, 0ull), impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        expDiff = tgmath::lerp(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                        rhsMantissa = tgmath::lerp(rhsMantissa | 0x4000000000000000ull, rhsMantissa, rhsBiasedExp == 0);
                        rhsMantissa = impl::unpackUint64(impl::shift64RightJamming(impl::packUint64(rhsMantissa), expDiff));
                        lhsMantissa |= 0x4000000000000000ull;
                        frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                        biasedExp = lhsBiasedExp;
                        --biasedExp;
                        return createPreserveBitPattern(impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 10, frac.x, frac.y));
                    }
                    if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        bool propagate = ((lhsMantissa) | (rhsMantissa)) != 0u;
                        return createPreserveBitPattern(tgmath::lerp(ieee754::traits<float64_t>::quietNaN, impl::propagateFloat64NaN(data, rhs.data), propagate));
                    }
                    rhsBiasedExp = tgmath::lerp(rhsBiasedExp, 1, lhsBiasedExp == 0);
                    lhsBiasedExp = tgmath::lerp(lhsBiasedExp, 1, lhsBiasedExp == 0);


                    const uint32_t2 lhsMantissaPacked = impl::packUint64(lhsMantissa);
                    const uint32_t2 rhsMantissaPacked = impl::packUint64(rhsMantissa);

                    uint32_t2 frac;
                    uint64_t signOfDifference = 0;
                    if (rhsMantissaPacked.x < lhsMantissaPacked.x)
                    {
                        frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                    }
                    else if (lhsMantissaPacked.x < rhsMantissaPacked.x)
                    {
                        frac.xy = impl::packUint64(rhsMantissa - lhsMantissa);
                        signOfDifference = ieee754::traits<float64_t>::signMask;
                    }
                    else if (rhsMantissaPacked.y <= lhsMantissaPacked.y)
                    {
                        /* It is possible that frac.x and frac.y may be zero after this. */
                        frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                    }
                    else
                    {
                        frac.xy = impl::packUint64(rhsMantissa - lhsMantissa);
                        signOfDifference = ieee754::traits<float64_t>::signMask;
                    }

                    biasedExp = tgmath::lerp(rhsBiasedExp, lhsBiasedExp, signOfDifference == 0u);
                    lhsSign ^= signOfDifference;
                    uint64_t retval_0 = impl::packFloat64(uint32_t(FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) << 31, 0, 0u, 0u);
                    uint64_t retval_1 = impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 11, frac.x, frac.y);
                    return createPreserveBitPattern(tgmath::lerp(retval_0, retval_1, frac.x != 0u || frac.y != 0u));
                }
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return createPreserveBitPattern(0xdeadbeefbadcaffeull);
            }
        }

        emulated_float64_t operator+(float rhs)
        {
            return createPreserveBitPattern(data) + create(rhs);
        }

        emulated_float64_t operator-(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            emulated_float64_t lhs = createPreserveBitPattern(data);
            emulated_float64_t rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
        }

        emulated_float64_t operator-(float rhs) NBL_CONST_MEMBER_FUNC
        {
            return createPreserveBitPattern(data) - create(rhs);
        }

        emulated_float64_t operator*(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if(FlushDenormToZero)
            {
                emulated_float64_t retval = emulated_float64_t::createPreserveBitPattern(0u);

                uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
                uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
                uint64_t lhsMantissa = ieee754::extractMantissa(data);
                uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                int exp = int(lhsBiasedExp + rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
                uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;
                if (!FastMath)
                {
                    if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        if ((lhsMantissa != 0u) || ((rhsBiasedExp == ieee754::traits<float64_t>::specialValueExp) && (rhsMantissa != 0u)))
                            return createPreserveBitPattern(impl::propagateFloat64NaN(data, rhs.data));
                        if ((uint64_t(rhsBiasedExp) | rhsMantissa) == 0u)
                            return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN);

                        return createPreserveBitPattern(impl::assembleFloat64(sign, ieee754::traits<float64_t>::exponentMask, 0ull));
                    }
                    if (rhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        /* a cannot be NaN, but is b NaN? */
                        if (rhsMantissa != 0u)
#ifdef RELAXED_NAN_PROPAGATION
                            return rhs.data;
#else
                            return createPreserveBitPattern(impl::propagateFloat64NaN(data, rhs.data));
#endif
                        if ((uint64_t(lhsBiasedExp) | lhsMantissa) == 0u)
                            return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN);

                        return createPreserveBitPattern(sign | ieee754::traits<float64_t>::exponentMask);
                    }
                    if (lhsBiasedExp == 0)
                    {
                        if (lhsMantissa == 0u)
                            return createPreserveBitPattern(sign);
                        impl::normalizeFloat64Subnormal(lhsMantissa, lhsBiasedExp, lhsMantissa);
                    }
                    if (rhsBiasedExp == 0)
                    {
                        if (rhsMantissa == 0u)
                            return createPreserveBitPattern(sign);
                        impl::normalizeFloat64Subnormal(rhsMantissa, rhsBiasedExp, rhsMantissa);
                    }
                }

                const uint64_t hi_l = (lhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_l = lhsMantissa & ((1ull << 21) - 1);
                const uint64_t hi_r = (rhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_r = rhsMantissa & ((1ull << 21) - 1);

                //const uint64_t RoundToNearest = (1ull << 31) - 1;
                uint64_t newPseudoMantissa = ((hi_l * hi_r) >> 10) + ((hi_l * lo_r + lo_l * hi_r/* + RoundToNearest*/) >> 31);

                if (newPseudoMantissa & (0x1ull << 53))
                {
                    newPseudoMantissa >>= 1;
                    ++exp;
                }
                newPseudoMantissa &= (ieee754::traits<float64_t>::mantissaMask);

                return createPreserveBitPattern(impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, newPseudoMantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return createPreserveBitPattern(0xdeadbeefbadcaffeull);
            }
        }

        emulated_float64_t operator*(float rhs)
        {
            return createPreserveBitPattern(data) * create(rhs);
        }

        /*emulated_float64_t<FastMath, FlushDenormToZero> reciprocal(uint64_t x)
        {
            using ThisType = emulated_float64_t<FastMath, FlushDenormToZero>;
            ThisType output = ThisType::createPreserveBitPattern((0xbfcdd6a18f6a6f52ULL - x) >> 1);
            output = output * output;
            return output;
        }*/

        emulated_float64_t operator/(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FlushDenormToZero)
            {
                //return emulated_float64_t<FastMath, FlushDenormToZero>::createPreserveBitPattern(data) * reciprocal(rhs.data);

                if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                    return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN);
                if (!FastMath && ((rhs.data << 1) == 0))
                    return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN);

                const uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;

                if (!FastMath && impl::areBothInfinity(data, rhs.data))
                    return createPreserveBitPattern(ieee754::traits<float64_t>::quietNaN | sign);

                if (!FastMath && tgmath::isInf(data))
                    return createPreserveBitPattern((data & ~ieee754::traits<float64_t>::signMask) | sign);

                if (!FastMath && tgmath::isInf(rhs.data))
                    return createPreserveBitPattern(0ull | sign);



                const uint64_t lhsRealMantissa = (ieee754::extractMantissa(data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt));
                const uint64_t rhsRealMantissa = ieee754::extractMantissa(rhs.data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);

                int exp = ieee754::extractExponent(data) - ieee754::extractExponent(rhs.data) + ieee754::traits<float64_t>::exponentBias;

                uint64_t2 lhsMantissaShifted = impl::shiftMantissaLeftBy53(lhsRealMantissa);
                uint64_t mantissa = impl::divmod128by64(lhsMantissaShifted.x, lhsMantissaShifted.y, rhsRealMantissa);

                while (mantissa < (1ull << 52))
                {
                    mantissa <<= 1;
                    exp--;
                }

                mantissa &= ieee754::traits<float64_t>::mantissaMask;

                return createPreserveBitPattern(impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, mantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return createPreserveBitPattern(0xdeadbeefbadcaffeull);
            }
        }

        // relational operators
        // TODO: should `FlushDenormToZero` affect relational operators?
        bool operator==(emulated_float64_t<FastMath, FlushDenormToZero> rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;
            // TODO: i'm not sure about this one
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return true;

            const emulated_float64_t xored = createPreserveBitPattern(data ^ rhs.data);
            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return true;

            return !(xored.data);
        }
        bool operator!=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            return !(createPreserveBitPattern(data) == rhs);
        }
        bool operator<(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;
            if (!FastMath && impl::areBothSameSignInfinity(data, rhs.data))
                return false;
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return false;

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) < (rhsFlipped & diffBits);
        }
        bool operator>(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return true;
            if (!FastMath && impl::areBothSameSignInfinity(data, rhs.data))
                return false;
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return false;

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) > (rhsFlipped & diffBits);
        }
        bool operator<=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC { return !(emulated_float64_t::createPreserveBitPattern(data) > emulated_float64_t::createPreserveBitPattern(rhs.data)); }
        bool operator>=(emulated_float64_t rhs) { return !(emulated_float64_t::createPreserveBitPattern(data) < emulated_float64_t::createPreserveBitPattern(rhs.data)); }

        //logical operators
        bool operator&&(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC { return bool(data) && bool(rhs.data); }
        bool operator||(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC { return bool(data) || bool(rhs.data); }
        bool operator!() NBL_CONST_MEMBER_FUNC { return !bool(data); }
        
        // TODO: should modify self?
        emulated_float64_t flipSign()
        {
            return createPreserveBitPattern(data ^ ieee754::traits<float64_t>::signMask);
        }
        
        bool isNaN()
        {
            return tgmath::isnan(data);
        }
    };

#define COMMA ,
#define IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(Type) \
template<>\
struct traits_base<Type >\
{\
    NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 11;\
    NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 52;\
};\
template<>\
inline uint32_t extractBiasedExponent(Type x)\
{\
    return extractBiasedExponent<uint64_t>(x.data);\
}\
\
template<>\
inline int extractExponent(Type x)\
{\
    return extractExponent(x.data);\
}\
\
template<>\
NBL_CONSTEXPR_INLINE_FUNC Type replaceBiasedExponent(Type x, typename unsigned_integer_of_size<sizeof(Type)>::type biasedExp)\
{\
    return Type(replaceBiasedExponent(x.data, biasedExp));\
}\
\
template <>\
NBL_CONSTEXPR_INLINE_FUNC Type fastMulExp2(Type x, int n)\
{\
    return Type(replaceBiasedExponent(x.data, extractBiasedExponent(x) + uint32_t(n)));\
}\
\
template <>\
NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<sizeof(Type)>::type extractMantissa(Type x)\
{\
    return extractMantissa(x.data);\
}\

namespace ieee754
{
    IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<true COMMA true>);
    IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<false COMMA false>);
    IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<true COMMA false>);
    IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<false COMMA true>);
}

}
}

#undef FLOAT_ROUND_NEAREST_EVEN
#undef FLOAT_ROUND_TO_ZERO
#undef FLOAT_ROUND_DOWN
#undef FLOAT_ROUND_UP
#undef FLOAT_ROUNDING_MODE

#undef COMMA
#undef IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE

#endif
