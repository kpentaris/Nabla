#ifndef _IRR_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/cos_weighted_sample.glsl>

float irr_glsl_lambertian_transmitter()
{
    return irr_glsl_RECIPROCAL_PI*0.5;
}

float irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(in float abs00NdotL)
{
   return absNdotL;
}
float irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out(in float NdotL)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(abs(NdotL));
}

float irr_glsl_lambertian_transmitter_cos_eval_wo_clamps(in irr_glsl_BSDFIsotropicParams params)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(params.NdotL)*irr_glsl_lambertian_transmitter();
}
float irr_glsl_lambertian_transmitter_cos_eval(in irr_glsl_BSDFIsotropicParams params)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out(params.NdotL)*irr_glsl_lambertian_transmitter();
}

irr_glsl_BSDFSample irr_glsl_lambertian_transmitter_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u)
{
    vec3 L = irr_glsl_projected_sphere_generate(u);

    irr_glsl_BSDFSample s;
    s.L = irr_glsl_getTangentFrame(interaction) * L;
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    /* Undefined
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;*/

    return s;
}

float irr_glsl_lambertian_transmitter_cos_remainder_and_pdf_wo_clamps(out float pdf, in float NdotL)
{
    return irr_glsl_projected_sphere_remainder_and_pdf(pdf,NdotL);
}
float irr_glsl_lambertian_transmitter_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s)
{
    return irr_glsl_lambertian_transmitter_cos_remainder_and_pdf_wo_clamps(pdf,abs(s.NdotL));
}

#endif
