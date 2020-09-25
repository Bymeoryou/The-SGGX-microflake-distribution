#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SymmetricGGXSpecular final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_RENDER_BASIC_TYPES()
    MTS_IMPORT_TYPES(Texture)

    SymmetricGGXSpecular(const Properties &props) : Base(props) {
        m_roughness = props.texture<Texture>("roughness", 0.5f);
        m_flags     = BSDFFlags::Anisotropic | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);
    }

    void calc_matrix(Float &Sxx, Float &Syy, Float &Szz, Float &Sxy, Float &Sxz,
                     Float &Syz, Float roughness, Vector3f &omega3) const {
        Float roughness2 = roughness * roughness;
        Sxx = roughness2 * omega3[0] * omega3[0] + omega3[1] * omega3[1] +
              omega3[2] * omega3[2];
        Sxy = roughness2 * omega3[0] * omega3[1] - omega3[0] * omega3[1];
        Sxz = roughness2 * omega3[0] * omega3[2] - omega3[0] * omega3[2];
        Syy = roughness2 * omega3[1] * omega3[1] + omega3[0] * omega3[0] +
              omega3[2] * omega3[2];
        Szz = roughness2 * omega3[2] * omega3[2] + omega3[0] * omega3[0] +
              omega3[1] * omega3[1];
        Syz = roughness2 * omega3[1] * omega3[2] - omega3[1] * omega3[2];
    }

    Float sigma(const Vector3f &wi, Float Sxx, Float Syy, Float Szz, Float Sxy,
                Float Sxz, Float Syz) const {
        const Float sigma_squared =
            wi[0] * wi[0] * Sxx + wi[1] * wi[1] * Syy + wi[2] * wi[2] * Szz +
            2.0f * (wi[0] * wi[1] * Sxy + wi[0] * wi[2] * Sxz +
                    wi[1] * wi[2] * Syz);
        return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f;
    }

    Float D(const Vector3f &wm, Float S_xx, Float S_yy, Float S_zz, Float S_xy,
            Float S_xz, Float S_yz) const {
        const Float detS = S_xx * S_yy * S_zz - S_xx * S_yz * S_yz -
                           S_yy * S_xz * S_xz - S_zz * S_xy * S_xy +
                           2.0f * S_xy * S_xz * S_yz;
        const Float den = wm[0] * wm[0] * (S_yy * S_zz - S_yz * S_yz) +
                          wm[1] * wm[1] * (S_xx * S_zz - S_xz * S_xz) +
                          wm[2] * wm[2] * (S_xx * S_yy - S_xy * S_xy) +
                          2.0f * (wm[0] * wm[1] * (S_xz * S_yz - S_zz * S_xy) +
                                  wm[0] * wm[2] * (S_xy * S_yz - S_yy * S_xz) +
                                  wm[1] * wm[2] * (S_xy * S_xz - S_xx * S_yz));
        Float D = powf(fabsf(detS), 1.5f) / (M_PI * den * den);
        return D;
    }

    Float eval_specular(const Vector3f &wi, const Vector3f &wo, Float S_xx,
                        Float S_yy, Float S_zz, Float S_xy, Float S_xz,
                        Float S_yz) const {
        Vector wh = normalize(wi + wo);
        return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) /
               sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2,
           Mask active = true) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs   = zero<BSDFSample3f>();

        active &= cos_theta_i > 0.f;
        if (unlikely(none_or<false>(active) ||
                     !ctx.is_enabled(BSDFFlags::DiffuseReflection)))
            return { bs, 0.f };

        bs.wo                = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf               = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta               = 1.f;
        bs.sampled_type      = +BSDFFlags::DiffuseReflection;
        bs.sampled_component = 0;

        UnpolarizedSpectrum value = m_roughness->eval(si, active);

        return { bs, select(active && bs.pdf > 0.f,
                            unpolarized<Spectrum>(value), 0.f) };        
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::Anisotropic))
            return 0.f;

        Vector3f omega3 = si.sh_frame.t;
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        calc_matrix(Sxx, Syy, Szz, Sxy, Sxz, Syz, m_roughness, omega3);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        UnpolarizedSpectrum value =
            m_roughness->eval(si, active) *
            eval_specular(si.wi, wo, Sxx, Syy, Szz, Sxy, Sxz, Syz);

        return select(active, unpolarized<Spectrum>(value), 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::Anisotropic))
            return 0.f;
        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("roughness", m_roughness.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SymmetricGGXSpecular[" << std::endl
            << "   roughness = " << string::indent(m_roughness) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_roughness;
};

MTS_IMPLEMENT_CLASS_VARIANT(SymmetricGGXSpecular, BSDF)
MTS_EXPORT_PLUGIN(SymmetricGGXSpecular, "SGGX")
NAMESPACE_END(mitsuba)