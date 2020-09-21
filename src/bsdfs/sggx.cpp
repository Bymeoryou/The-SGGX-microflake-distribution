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
    MTS_IMPORT_TYPES(Texture)

    SymmetricGGXSpecular(const Properties &props) : Base(props) {
        m_roughness   = props.texture<Texture>("roughness", 0.5f);
        m_anisotropic = props.texture<Texture>("anisotropic", 0.6f);
        m_flags       = BSDFFlags::Anisotropic | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);
    }

    void calc_matrix(Float &Sxx, Float &Syy, Float Szz, Float &Sxy, Float &Sxz,
                     Float &Syz, Float roughness, const Vector3f &omega3) {
        Float roughness2 = roughness * roughness;
        Sxx = roughness2 * omega3.x * omega3.x + omega3.y * omega3.y +
              omega3.z * omega3.z;
        Sxy = roughness2 * omega3.x * omega3.y - omega3.x * omega3.y;
        Sxz = roughness2 * omega3.x * omega3.z - omega3.x * omega3.z;
        Syy = roughness2 * omega3.y * omega3.y + omega3.x * omega3.x +
              omega3.z * omega3.z;
        Szz = roughness2 * omega3.z * omega3.z + omega3.x * omega3.x +
              omega3.y * omega3.y;
        Syz = roughness2 * omega3.y * omega3.z - omega3.y * omega3.z;
    }

    Float sigma(const Vector3f &wi, Float Sxx, Float Syy, Float Szz, Float Sxy,
                Float Sxz, Float Syz) {
        const Float sigma_squared =
            wi.x * wi.x * Sxx + wi.y * wi.y * Syy + wi.z * wi.z * Szz +
            2.0f * (wi.x * wi.y * Sxy + wi.x * wi.z * Sxz + wi.y * wi.z * Syz);
        return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f;
    }

    Float D(const Vector3f &wm, Float S_xx, Float S_yy, Float S_zz, Float S_xy,
            Float S_xz, Float S_yz) {
        const Float detS = S_xx * S_yy * S_zz - S_xx * S_yz * S_yz -
                           S_yy * S_xz * S_xz - S_zz * S_xy * S_xy +
                           2.0f * S_xy * S_xz * S_yz;
        const Float den = wm.x * wm.x * (S_yy * S_zz - S_yz * S_yz) +
                          wm.y * wm.y * (S_xx * S_zz - S_xz * S_xz) +
                          wm.z * wm.z * (S_xx * S_yy - S_xy * S_xy) +
                          2.0f * (wm.x * wm.y * (S_xz * S_yz - S_zz * S_xy) +
                                  wm.x * wm.z * (S_xy * S_yz - S_yy * S_xz) +
                                  wm.y * wm.z * (S_xy * S_xz - S_xx * S_yz));
        const Float D = powf(fabsf(detS), 1.5f) / (M_PI * den * den);
        return D;
    }

    void buildOrthonormalBasis(Vector3f &wk, Vector3f &wj, const Vector3f &wi) {
        if (wi.z < -0.9999999f) {
            wk = Vector(0.0f, -1.0f, 0.0f);
            wj = Vector(-1.0f, 0.0f, 0.0f);
        } else {
            const Float a = 1.0f / (1.0f + wi.z);
            const Float b = -wi.x * wi.y * a;
            wk            = Vector(1.0f - wi.x * wi.x * a, b, -wi.x);
            wj            = Vector(b, 1.0f - wi.y * wi.y * a, -wi.y);
        }
    }

    Vector3f sample_VNDF(const Vector3f &wi, Float S_xx, Float S_yy, Float S_zz,
                       Float S_xy, Float S_xz, Float S_yz, Float U1, Float U2) {
        // generate sample (u, v, w)
        const Float r   = sqrtf(U1);
        const Float phi = 2.0f * M_PI * U2;
        const Float u   = r * cosf(phi);
        const Float v   = r * sinf(phi);
        const Float w   = sqrtf(1.0f - u * u - v * v);

        // build orthonormal basis
        Vector wk, wj;
        buildOrthonormalBasis(wk, wj, wi);
        // project S in this basis
        const Float S_kk =
            wk.x * wk.x * S_xx + wk.y * wk.y * S_yy + wk.z * wk.z * S_zz +
            2.0f *
                (wk.x * wk.y * S_xy + wk.x * wk.z * S_xz + wk.y * wk.z * S_yz);
        const Float S_jj =
            wj.x * wj.x * S_xx + wj.y * wj.y * S_yy + wj.z * wj.z * S_zz +
            2.0f *
                (wj.x * wj.y * S_xy + wj.x * wj.z * S_xz + wj.y * wj.z * S_yz);
        const Float S_ii =
            wi.x * wi.x * S_xx + wi.y * wi.y * S_yy + wi.z * wi.z * S_zz +
            2.0f *
                (wi.x * wi.y * S_xy + wi.x * wi.z * S_xz + wi.y * wi.z * S_yz);
        const Float S_kj = wk.x * wj.x * S_xx + wk.y * wj.y * S_yy +
                           wk.z * wj.z * S_zz +
                           (wk.x * wj.y + wk.y * wj.x) * S_xy +
                           (wk.x * wj.z + wk.z * wj.x) * S_xz +
                           (wk.y * wj.z + wk.z * wj.y) * S_yz;
        const Float S_ki = wk.x * wi.x * S_xx + wk.y * wi.y * S_yy +
                           wk.z * wi.z * S_zz +
                           (wk.x * wi.y + wk.y * wi.x) * S_xy +
                           (wk.x * wi.z + wk.z * wi.x) * S_xz +
                           (wk.y * wi.z + wk.z * wi.y) * S_yz;
        const Float S_ji = wj.x * wi.x * S_xx + wj.y * wi.y * S_yy +
                           wj.z * wi.z * S_zz +
                           (wj.x * wi.y + wj.y * wi.x) * S_xy +
                           (wj.x * wi.z + wj.z * wi.x) * S_xz +
                           (wj.y * wi.z + wj.z * wi.y) * S_yz;
        // compute normal
        Float sqrtDetSkji = sqrtf(
            fabsf(S_kk * S_jj * S_ii - S_kj * S_kj * S_ii - S_ki * S_ki * S_jj -
                  S_ji * S_ji * S_kk + 2.0f * S_kj * S_ki * S_ji));
        Float inv_sqrtS_ii = 1.0f / sqrtf(S_ii);
        Float tmp          = sqrtf(S_jj * S_ii - S_ji * S_ji);
        Vector Mk(sqrtDetSkji / tmp, 0.0f, 0.0f);
        Vector Mj(-inv_sqrtS_ii * (S_ki * S_ji - S_kj * S_ii) / tmp,
                  inv_sqrtS_ii * tmp, 0);
        Vector Mi(inv_sqrtS_ii * S_ki, inv_sqrtS_ii * S_ji,
                  inv_sqrtS_ii * S_ii);
        Vector wm_kji = normalize(u * Mk + v * Mj + w * Mi);
        // rotate back to world basis
        return wm_kji.x * wk + wm_kji.y * wj + wm_kji.z * wi;
    }

    Float eval_specular(const Vector3f &wi, const Vector3f &wo, Float S_xx,
                        Float S_yy, Float S_zz, Float S_xy, Float S_xz,
                        Float S_yz) {
        Vector wh = normalize(wi + wo);
        return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) /
               sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
    }

    Vector3f sample_specular(const Vector3f &wi, Float S_xx, Float S_yy, Float S_zz,
                           Float S_xy, Float S_xz, Float S_yz, Float U1,
                           Float U2) {
        // sample VNDF
        const Vector wm =
            sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // specular reflection
        const Vector wo = -wi + 2.0f * wm * dot(wm, wi);
        return wo;
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Vector omega3 = si.sh_frame.t;
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


    void traverse(TraversalCallback *callback) override {
        callback->put_object("roughness", m_roughness.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SymmetricGGXSpecular[" << std::endl
            << "   roughness = " << m_roughness << std::endl
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