#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/warp.h>

#include <mitsuba/render/texture.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>


NAMESPACE_BEGIN(mitsuba)


template <typename Float, typename Spectrum>
class SymmetricGGXSpecular final : public PhaseFunction<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)
    
    SymmetricGGXSpecular(const Properties &props) : Base(props) {
        m_roughness = props.texture<Texture>("roughness", 0.5f);
        m_anisotropic = props.texture<Texture>("anisotropic", 0.6f);
        m_flags = PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    void calc_matrix(Float &Sxx, Float &Syy, Float Szz, Float &Sxy, Float &Sxz,
                     Float &Syz, Float roughness, const Vector &omega3) {
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

    Float sigma(const Vector &wi, Float Sxx, Float Syy, Float Szz, Float Sxy,
                Float Sxz, Float Syz) {
        const Float sigma_squared =
            wi.x * wi.x * Sxx + wi.y * wi.y * Syy + wi.z * wi.z * Szz +
            2.0f * (wi.x * wi.y * Sxy + wi.x * wi.z * Sxz + wi.y * wi.z * Syz);
        return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f;
    }

    Float D(const Vector &wm, Float S_xx, Float S_yy, Float S_zz, Float S_xy,
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

    void buildOrthonormalBasis(Vector &omega_1, Vector &omega_2,
                               const Vector &omega_3) {
        if (omega_3.z < -0.9999999f) {
            omega_1 = Vector(0.0f, -1.0f, 0.0f);
            omega_2 = Vector(-1.0f, 0.0f, 0.0f);
        } else {
            const Float a = 1.0f / (1.0f + omega_3.z);
            const Float b = -omega_3.x * omega_3.y * a;
            omega_1 = Vector(1.0f - omega_3.x * omega_3.x * a, b, -omega_3.x);
            omega_2 = Vector(b, 1.0f - omega_3.y * omega_3.y * a, -omega_3.y);
        }
    }

    Vector sample_VNDF(const Vector &wi, Float S_xx, Float S_yy, Float S_zz,
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

    Float eval_specular(const Vector &wi, const Vector &wo, Float S_xx,
                        Float S_yy, Float S_zz, Float S_xy, Float S_xz,
                        Float S_yz) {
        Vector wh = normalize(wi + wo);
        return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) /
               sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
    }

    Vector sample_specular(const Vector &wi, Float S_xx, Float S_yy, Float S_zz,
                           Float S_xy, Float S_xz, Float S_yz, Float U1,
                           Float U2) {
        // sample VNDF
        const Vector wm =
            sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // specular reflection
        const Vector wo = -wi + 2.0f * wm * dot(wm, wi);
        return wo;
    }

    Float eval_diffuse(const Vector &wi, const Vector &wo, Float S_xx,
                       Float S_yy, Float S_zz, Float S_xy, Float S_xz,
                       Float S_yz, Float U1, Float U2) {
        // sample VNDF
        const Vector wm =
            sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // eval diffuse
        return 1.0f / M_PI * fmaxf(0.0f, dot(wo, wm));
    }

    Vector sample_diffuse(const Vector &wi, Float S_xx, Float S_yy, Float S_zz,
                          Float S_xy, Float S_xz, Float S_yz, Float U1,
                          Float U2, Float U3, Float U4) {
        // sample VNDF
        const Vector wm =
            sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // sample diffuse reflection
        Vector w1, w2;
        buildOrthonormalBasis(w1, w2, wm);
        Float r1 = 2.0f * U3 - 1.0f;
        Float r2 = 2.0f * U4 - 1.0f;
        // concentric map code from
        // http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
        Float phi, r;
        if (r1 == 0 && r2 == 0) {
            r = phi = 0;
        } else if (r1 * r1 > r2 * r2) {
            r   = r1;
            phi = (M_PI / 4.0f) * (r2 / r1);
        } else {
            r   = r2;
            phi = (M_PI / 2.0f) - (r1 / r2) * (M_PI / 4.0f);
        }
        Float x   = r * cosf(phi);
        Float y   = r * sinf(phi);
        Float z   = sqrtf(1.0f - x * x - y * y);
        Vector wo = x * w1 + y * w2 + z * wm;
        return wo;
    }

    Spectrum eval(const PhaseFunctionContext& ctx, const MediumInteraction3f& mi,
        const Vector3f& wo, Mask active)  const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Vector omega3 = mi.t;
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        calc_matrix(Sxx, Syy, Szz, Sxy, Sxz, Syz);

        Vector wi = mi.to_local(mi.wi);
        wo = mi.to_local(wo);
        Float ret = eval_specular(wi, wo, Sxxx, Syy, Szz, Sxy, Sxz, Syz);

        return ret;
    }

    std::pair<Vector3f, Float> sample(const PhaseFunctionContext& ctx,
        const MediumInteraction3f& mi, const Point2f& sample, Mask active = true) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);


    }

private:
    ref<Texture> m_texture_roughness;
    ref<Texture> m_texture_anisotropic;

};


MTS_EXPORT_PLUGIN(SymmetricGGXSpecular, "SGGX")
NAMESPACE_END(mitsuba)