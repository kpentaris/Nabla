include_guard(GLOBAL)

option(NBL_SANITIZE_THREAD OFF)
option(NBL_REQUEST_F16C "Request compilation with F16C enabled for Nabla projects" ON)

# https://en.wikipedia.org/wiki/F16C
# support for converting between half-precision and standard IEEE single-precision floating-point formats
if(NBL_REQUEST_F16C)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT("-mf16c")
endif()

# Debug
set(NBL_C_DEBUG_COMPILE_OPTIONS
	-Wall -fno-omit-frame-pointer -fstack-protector-strong
)
set(NBL_CXX_DEBUG_COMPILE_OPTIONS
	${NBL_C_DEBUG_COMPILE_OPTIONS}
)

set(NBL_DEBUG_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_DEBUG_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_DEBUG_COMPILE_OPTIONS}>
PARENT_SCOPE)

# Release
set(NBL_C_RELEASE_COMPILE_OPTIONS
	# empty
)

if(FAST_MATH)
	list(APPEND -ffast-math -ffast-math)
endif()

set(NBL_CXX_RELEASE_COMPILE_OPTIONS
	${NBL_C_RELEASE_COMPILE_OPTIONS}
)

set(NBL_RELEASE_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_RELEASE_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_RELEASE_COMPILE_OPTIONS}>
PARENT_SCOPE)

# RelWithDebInfo
set(NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS
	# empty
)
set(NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS
	${NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS}
)

set(NBL_RELWITHDEBINFO_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS}>
PARENT_SCOPE)

# Global
unset(NBL_C_COMPILE_OPTIONS)
unset(NBL_CXX_COMPILE_OPTIONS)

list(APPEND NBL_C_COMPILE_OPTIONS
	-Wextra
	-Wno-unused-parameter
	-fno-strict-aliasing
	-msse4.2
	-mfpmath=sse		
	-Wextra
	-Wno-sequence-point
	-Wno-error=ignored-attributes
	-Wno-error=unused-function
	-Wno-error=unused-variable
	-Wno-error=unused-parameter
	-fno-exceptions
	$<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,C>:-maes>
)

if(NBL_SANITIZE_ADDRESS)
	list(APPEND NBL_C_COMPILE_OPTIONS -fsanitize=address)
endif()

if(NBL_SANITIZE_THREAD)
	list(APPEND NBL_C_COMPILE_OPTIONS -fsanitize=thread)
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.1)
	list(APPEND NBL_C_COMPILE_OPTIONS -Wno-error=ignored-attributes)
endif()

list(APPEND NBL_CXX_COMPILE_OPTIONS ${NBL_C_COMPILE_OPTIONS})

set(NBL_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_COMPILE_OPTIONS}>
PARENT_SCOPE)