include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_IMGUI_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/ImGui/ImGui.h
)

set(NBL_EXT_IMGUI_SRC
	ImGui.cpp
)

set(NBL_EXT_IMGUI_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_EXT_INTERNAL_INCLUDE_DIR}"
)

nbl_create_ext_library_project(
	IMGUI_UI
	"${NBL_EXT_IMGUI_H}"
	"${NBL_EXT_IMGUI_SRC}"
	"${NBL_EXT_IMGUI_EXTERNAL_INCLUDE}"
	""
    ""
)

# shaders IO directories
set(NBL_EXT_IMGUI_INPUT_SHADERS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
get_filename_component(_EXT_IMGUI_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_BINARY_DIR}/shaders/include" ABSOLUTE)
get_filename_component(_EXT_IMGUI_SPIRV_BR_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/include" ABSOLUTE)
get_filename_component(_EXT_IMGUI_SPIRV_BR_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/src" ABSOLUTE)
set(NBL_EXT_IMGUI_OUTPUT_SPIRV_DIRECTORY "${_EXT_IMGUI_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_}/nbl/ext/imgui/spirv")

# list of input source shaders
set(NBL_EXT_IMGUI_INPUT_SHADERS
	"${NBL_EXT_IMGUI_INPUT_SHADERS_DIRECTORY}/vertex.hlsl"
	"${NBL_EXT_IMGUI_INPUT_SHADERS_DIRECTORY}/fragment.hlsl"
)

set(NBL_EXT_IMGUI_INPUT_COMMONS
	"${NBL_EXT_IMGUI_INPUT_SHADERS_DIRECTORY}/common.hlsl"
)

include("${NBL_ROOT_PATH}/src/nbl/builtin/utils.cmake")

foreach(NBL_INPUT_SHADER IN LISTS NBL_EXT_IMGUI_INPUT_SHADERS)
	cmake_path(GET NBL_INPUT_SHADER STEM NBL_SHADER_STEM)
	set(NBL_OUTPUT_SPIRV_FILENAME "${NBL_SHADER_STEM}.spv")
	set(NBL_OUTPUT_SPIRV_PATH "${NBL_EXT_IMGUI_OUTPUT_SPIRV_DIRECTORY}/${NBL_OUTPUT_SPIRV_FILENAME}")

	if(NBL_SHADER_STEM STREQUAL vertex)
		set(NBL_NSC_COMPILE_OPTIONS -T vs_6_7 -E VSMain)
	elseif(NBL_SHADER_STEM STREQUAL fragment)
		set(NBL_NSC_COMPILE_OPTIONS -T ps_6_7 -E PSMain)
	else()
		message(FATAL_ERROR "internal error")
	endif()
	
	set(NBL_NSC_COMPILE_COMMAND
		"$<TARGET_FILE:nsc>"
		-Fc "${NBL_OUTPUT_SPIRV_PATH}"
		${NBL_NSC_COMPILE_OPTIONS} # this should come from shader's [#pragma WAVE <compile options>] but our NSC doesn't seem to work properly currently
		"${NBL_INPUT_SHADER}"
	)
	
	set(NBL_DEPENDS
		"${NBL_INPUT_SHADER}"
		${NBL_EXT_IMGUI_INPUT_COMMONS}
	)
		
	add_custom_command(OUTPUT "${NBL_OUTPUT_SPIRV_PATH}"
	   COMMAND ${NBL_NSC_COMPILE_COMMAND}
	   DEPENDS ${NBL_DEPENDS}
	   WORKING_DIRECTORY "${NBL_EXT_IMGUI_INPUT_SHADERS_DIRECTORY}"
	   COMMENT "Generating \"${NBL_OUTPUT_SPIRV_PATH}\""
	   VERBATIM
	   COMMAND_EXPAND_LISTS
	)
	
	list(APPEND NBL_EXT_IMGUI_OUTPUT_SPIRV_BUILTINS "${NBL_OUTPUT_SPIRV_PATH}")
	LIST_BUILTIN_RESOURCE(IMGUI_EXT_SPIRV_RESOURCES_TO_EMBED "ext/imgui/spirv/${NBL_OUTPUT_SPIRV_FILENAME}")
endforeach()

ADD_CUSTOM_BUILTIN_RESOURCES(extImguiSpirvBuiltinResourceData IMGUI_EXT_SPIRV_RESOURCES_TO_EMBED "${_EXT_IMGUI_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_}" "nbl" "ext::imgui::spirv::builtin" "${_EXT_IMGUI_SPIRV_BR_OUTPUT_DIRECTORY_HEADER_}" "${_EXT_IMGUI_SPIRV_BR_OUTPUT_DIRECTORY_SOURCE_}" "STATIC" "INTERNAL")
LINK_BUILTIN_RESOURCES_TO_TARGET(${LIB_NAME} extImguiSpirvBuiltinResourceData)