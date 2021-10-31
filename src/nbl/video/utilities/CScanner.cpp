#include "nbl/video/utilities/CScanner.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<asset::ICPUShader> CScanner::createShader(const bool indirect, const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op) const
{
	auto system = m_device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
	if (indirect)
		glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/scan/indirect.comp")>();
	else
		glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/scan/direct.comp")>();
	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl),asset::ICPUShader::buffer_contains_glsl, asset::IShader::ESS_COMPUTE, "nbl/builtin/glsl/scan/default.comp");
	const char* storageType = nullptr;
	switch (dataType)
	{
		case EDT_UINT:
			storageType = "uint";
			break;
		case EDT_INT:
			storageType = "int";
			break;
		case EDT_FLOAT:
			storageType = "float";
			break;
		default:
			assert(false);
			break;
	}
	return asset::IGLSLCompiler::createOverridenCopy(
		cpushader.get(),
		"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_GLSL_WORKGROUP_SIZE_LOG2_ %d\n#define _NBL_GLSL_SCAN_TYPE_ %d\n#define _NBL_GLSL_SCAN_STORAGE_TYPE_ %s\n#define _NBL_GLSL_SCAN_BIN_OP_ %d\n",
		m_wg_size,core::findMSB(m_wg_size),uint32_t(scanType),storageType,uint32_t(op)
	);
}