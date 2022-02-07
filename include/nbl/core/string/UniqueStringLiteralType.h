// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_UNIQUE_STRING_LITERAL_TYPE_H_INCLUDED__
#define __NBL_CORE_UNIQUE_STRING_LITERAL_TYPE_H_INCLUDED__

#include "nbl/macros.h"

namespace nbl
{
namespace core
{
template<char... chars>
struct CharParameterPackToStringLiteral
{
    _NBL_STATIC_INLINE_CONSTEXPR char value[] = {chars..., '\0'};
};

}
}

#define NBL_CORE_MAX_GET_CHAR_STRLEN 128

#define NBL_CORE_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define NBL_CORE_STRLEN(s) (sizeof(s) / sizeof(s[0]) - 1)

#define NBL_CORE_GET_CHAR(str, ii) ((NBL_CORE_MIN(ii, NBL_CORE_MAX_GET_CHAR_STRLEN - 1)) < NBL_CORE_STRLEN(str) ? str[ii] : 0)

#define NBL_CORE_STRING_TO_CHAR_PARAMETER_PACK(s) \
    NBL_CORE_GET_CHAR(s, 0),                      \
        NBL_CORE_GET_CHAR(s, 1),                  \
        NBL_CORE_GET_CHAR(s, 2),                  \
        NBL_CORE_GET_CHAR(s, 3),                  \
        NBL_CORE_GET_CHAR(s, 4),                  \
        NBL_CORE_GET_CHAR(s, 5),                  \
        NBL_CORE_GET_CHAR(s, 6),                  \
        NBL_CORE_GET_CHAR(s, 7),                  \
        NBL_CORE_GET_CHAR(s, 8),                  \
        NBL_CORE_GET_CHAR(s, 9),                  \
        NBL_CORE_GET_CHAR(s, 10),                 \
        NBL_CORE_GET_CHAR(s, 11),                 \
        NBL_CORE_GET_CHAR(s, 12),                 \
        NBL_CORE_GET_CHAR(s, 13),                 \
        NBL_CORE_GET_CHAR(s, 14),                 \
        NBL_CORE_GET_CHAR(s, 15),                 \
        NBL_CORE_GET_CHAR(s, 16),                 \
        NBL_CORE_GET_CHAR(s, 17),                 \
        NBL_CORE_GET_CHAR(s, 18),                 \
        NBL_CORE_GET_CHAR(s, 19),                 \
        NBL_CORE_GET_CHAR(s, 20),                 \
        NBL_CORE_GET_CHAR(s, 21),                 \
        NBL_CORE_GET_CHAR(s, 22),                 \
        NBL_CORE_GET_CHAR(s, 23),                 \
        NBL_CORE_GET_CHAR(s, 24),                 \
        NBL_CORE_GET_CHAR(s, 25),                 \
        NBL_CORE_GET_CHAR(s, 26),                 \
        NBL_CORE_GET_CHAR(s, 27),                 \
        NBL_CORE_GET_CHAR(s, 28),                 \
        NBL_CORE_GET_CHAR(s, 29),                 \
        NBL_CORE_GET_CHAR(s, 30),                 \
        NBL_CORE_GET_CHAR(s, 31),                 \
        NBL_CORE_GET_CHAR(s, 32),                 \
        NBL_CORE_GET_CHAR(s, 33),                 \
        NBL_CORE_GET_CHAR(s, 34),                 \
        NBL_CORE_GET_CHAR(s, 35),                 \
        NBL_CORE_GET_CHAR(s, 36),                 \
        NBL_CORE_GET_CHAR(s, 37),                 \
        NBL_CORE_GET_CHAR(s, 38),                 \
        NBL_CORE_GET_CHAR(s, 39),                 \
        NBL_CORE_GET_CHAR(s, 40),                 \
        NBL_CORE_GET_CHAR(s, 41),                 \
        NBL_CORE_GET_CHAR(s, 42),                 \
        NBL_CORE_GET_CHAR(s, 43),                 \
        NBL_CORE_GET_CHAR(s, 44),                 \
        NBL_CORE_GET_CHAR(s, 45),                 \
        NBL_CORE_GET_CHAR(s, 46),                 \
        NBL_CORE_GET_CHAR(s, 47),                 \
        NBL_CORE_GET_CHAR(s, 48),                 \
        NBL_CORE_GET_CHAR(s, 49),                 \
        NBL_CORE_GET_CHAR(s, 50),                 \
        NBL_CORE_GET_CHAR(s, 51),                 \
        NBL_CORE_GET_CHAR(s, 52),                 \
        NBL_CORE_GET_CHAR(s, 53),                 \
        NBL_CORE_GET_CHAR(s, 54),                 \
        NBL_CORE_GET_CHAR(s, 55),                 \
        NBL_CORE_GET_CHAR(s, 56),                 \
        NBL_CORE_GET_CHAR(s, 57),                 \
        NBL_CORE_GET_CHAR(s, 58),                 \
        NBL_CORE_GET_CHAR(s, 59),                 \
        NBL_CORE_GET_CHAR(s, 60),                 \
        NBL_CORE_GET_CHAR(s, 61),                 \
        NBL_CORE_GET_CHAR(s, 62),                 \
        NBL_CORE_GET_CHAR(s, 63),                 \
        NBL_CORE_GET_CHAR(s, 64),                 \
        NBL_CORE_GET_CHAR(s, 65),                 \
        NBL_CORE_GET_CHAR(s, 66),                 \
        NBL_CORE_GET_CHAR(s, 67),                 \
        NBL_CORE_GET_CHAR(s, 68),                 \
        NBL_CORE_GET_CHAR(s, 69),                 \
        NBL_CORE_GET_CHAR(s, 70),                 \
        NBL_CORE_GET_CHAR(s, 71),                 \
        NBL_CORE_GET_CHAR(s, 72),                 \
        NBL_CORE_GET_CHAR(s, 72),                 \
        NBL_CORE_GET_CHAR(s, 72),                 \
        NBL_CORE_GET_CHAR(s, 73),                 \
        NBL_CORE_GET_CHAR(s, 74),                 \
        NBL_CORE_GET_CHAR(s, 75),                 \
        NBL_CORE_GET_CHAR(s, 76),                 \
        NBL_CORE_GET_CHAR(s, 77),                 \
        NBL_CORE_GET_CHAR(s, 78),                 \
        NBL_CORE_GET_CHAR(s, 79),                 \
        NBL_CORE_GET_CHAR(s, 80),                 \
        NBL_CORE_GET_CHAR(s, 81),                 \
        NBL_CORE_GET_CHAR(s, 82),                 \
        NBL_CORE_GET_CHAR(s, 83),                 \
        NBL_CORE_GET_CHAR(s, 84),                 \
        NBL_CORE_GET_CHAR(s, 85),                 \
        NBL_CORE_GET_CHAR(s, 86),                 \
        NBL_CORE_GET_CHAR(s, 87),                 \
        NBL_CORE_GET_CHAR(s, 88),                 \
        NBL_CORE_GET_CHAR(s, 89),                 \
        NBL_CORE_GET_CHAR(s, 90),                 \
        NBL_CORE_GET_CHAR(s, 91),                 \
        NBL_CORE_GET_CHAR(s, 92),                 \
        NBL_CORE_GET_CHAR(s, 93),                 \
        NBL_CORE_GET_CHAR(s, 94),                 \
        NBL_CORE_GET_CHAR(s, 95),                 \
        NBL_CORE_GET_CHAR(s, 96),                 \
        NBL_CORE_GET_CHAR(s, 97),                 \
        NBL_CORE_GET_CHAR(s, 98),                 \
        NBL_CORE_GET_CHAR(s, 99),                 \
        NBL_CORE_GET_CHAR(s, 100),                \
        NBL_CORE_GET_CHAR(s, 101),                \
        NBL_CORE_GET_CHAR(s, 102),                \
        NBL_CORE_GET_CHAR(s, 103),                \
        NBL_CORE_GET_CHAR(s, 104),                \
        NBL_CORE_GET_CHAR(s, 105),                \
        NBL_CORE_GET_CHAR(s, 106),                \
        NBL_CORE_GET_CHAR(s, 107),                \
        NBL_CORE_GET_CHAR(s, 108),                \
        NBL_CORE_GET_CHAR(s, 109),                \
        NBL_CORE_GET_CHAR(s, 110),                \
        NBL_CORE_GET_CHAR(s, 111),                \
        NBL_CORE_GET_CHAR(s, 112),                \
        NBL_CORE_GET_CHAR(s, 113),                \
        NBL_CORE_GET_CHAR(s, 114),                \
        NBL_CORE_GET_CHAR(s, 115),                \
        NBL_CORE_GET_CHAR(s, 116),                \
        NBL_CORE_GET_CHAR(s, 117),                \
        NBL_CORE_GET_CHAR(s, 118),                \
        NBL_CORE_GET_CHAR(s, 119),                \
        NBL_CORE_GET_CHAR(s, 120),                \
        NBL_CORE_GET_CHAR(s, 121),                \
        NBL_CORE_GET_CHAR(s, 122),                \
        NBL_CORE_GET_CHAR(s, 123),                \
        NBL_CORE_GET_CHAR(s, 124),                \
        NBL_CORE_GET_CHAR(s, 125),                \
        NBL_CORE_GET_CHAR(s, 126),                \
        NBL_CORE_GET_CHAR(s, 127)

//
#define NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(s) nbl::core::CharParameterPackToStringLiteral<NBL_CORE_STRING_TO_CHAR_PARAMETER_PACK(s)>

#endif
