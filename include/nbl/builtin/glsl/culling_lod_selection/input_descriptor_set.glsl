#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET_GLSL_INCLUDED_


#ifndef NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET
#define NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET 1
#endif

#include <nbl/builtin/glsl/utils/indirect_commands.glsl>
#include <nbl/builtin/glsl/culling_lod_selection/dispatch_indirect_params.glsl>
#ifndef NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_BINDING 0
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_BINDING
) restrict coherent buffer DispatchIndirect
{
    nbl_glsl_culling_lod_selection_dispatch_indirect_params_t dispatchIndirect;
};
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_BINDING 1
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_BINDING
) restrict readonly buffer InstanceList
{
    uvec2 data[]; // <instanceGUID,lod_table_t offset>
} instanceList;
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_PVS_INSTANCES_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_PVS_INSTANCES_DESCRIPTOR_BINDING 2
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_PVS_INSTANCES_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_PVS_INSTANCES_DESCRIPTOR_QUALIFIERS buffer PVSInstances
{
    uvec2 data[];
} pvsInstances;
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_BINDING 3
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_QUALIFIERS buffer LoDDrawcallInclusiveCounts
{
    uint totalInstanceCountAfterCull; // cleared by scatter
    uint lodDrawcallInclusiveCounts[];
};
#endif

struct nbl_glsl_culling_lod_selection_PotentiallyVisibleInstanceDraw_t
{
    uint perViewPerInstanceID;
    uint drawBaseInstanceDWORDOffset;
    uint instanceID;
}; // TODO: move
#ifndef NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_BINDING 4
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_QUALIFIERS buffer PVSInstanceDraws
{
    uint count; // cleared by LoD selection
    nbl_glsl_culling_lod_selection_PotentiallyVisibleInstanceDraw_t data[];
} pvsInstanceDraws;
#endif

// override the scan descriptors a bit
#ifndef _NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_ NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET
#endif
// we provide our own scan data
#define _NBL_GLSL_SCAN_INPUT_DESCRIPTOR_DEFINED_
// rearrange scratch binding a bit
#ifndef _NBL_GLSL_SCAN_SCRATCH_BINDING_DEFINED_
#define _NBL_GLSL_SCAN_SCRATCH_BINDING_DEFINED_ 5
#endif
// we will define these ourselves, but differently for different scans
#define _NBL_GLSL_SCAN_GET_PADDED_DATA_DEFINED_
#define _NBL_GLSL_SCAN_SET_DATA_DEFINED_
#include <nbl/builtin/glsl/scan/descriptors.glsl>

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_DRAWCALLS_TO_SCAN_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAWCALLS_TO_SCAN_DESCRIPTOR_BINDING 6
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_DRAWCALLS_TO_SCAN_DESCRIPTOR_BINDING
) restrict readonly buffer DrawcallsToScan
{
    uint dwordOffsets[];
} drawcallsToScan;
#endif

// TODO: do we even need this?
#ifndef NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_BINDING 7
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_INPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_BINDING
) restrict readonly buffer DrawCountsToScan
{
    uint data[];
} drawCountsToScan;
#endif

#endif