include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")

# Debug
list(APPEND NBL_C_DEBUG_COMPILE_OPTIONS
	-ggdb3
)
list(APPEND NBL_CXX_DEBUG_COMPILE_OPTIONS
	${NBL_C_DEBUG_COMPILE_OPTIONS}
)

set(NBL_DEBUG_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_DEBUG_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_DEBUG_COMPILE_OPTIONS}>
)

# Release
list(APPEND NBL_C_RELEASE_COMPILE_OPTIONS
	-fexpensive-optimizations
)
list(APPEND NBL_CXX_RELEASE_COMPILE_OPTIONS
	${NBL_C_RELEASE_COMPILE_OPTIONS}
)

set(NBL_RELEASE_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_RELEASE_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_RELEASE_COMPILE_OPTIONS}>
)

# RelWithDebInfo
list(APPEND NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS
	# empty
)
list(APPEND NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS
	${NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS}
)

set(NBL_RELWITHDEBINFO_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS}>
)

# Global
list(APPEND NBL_C_COMPILE_OPTIONS 
	-Wno-unused-but-set-parameter
	-fuse-ld=gold
)

list(APPEND NBL_CXX_COMPILE_OPTIONS ${NBL_C_COMPILE_OPTIONS})

set(NBL_COMPILE_OPTIONS
	$<$<COMPILE_LANGUAGE:CXX>:${NBL_CXX_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:C>:${NBL_C_COMPILE_OPTIONS}>
)

# our pervious flags-set function called this, does not affect flags nor configs so I will keep it here temporary
# TODO: move it out from the profile
link_libraries(-fuse-ld=gold)