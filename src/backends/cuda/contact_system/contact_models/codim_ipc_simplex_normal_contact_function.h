#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda
{
enum class SimplexBarrierModel : IndexT
{
    IPC = 0,
    Quadratic = 1,
};

namespace sym::codim_ipc_simplex_contact
{
    inline __device__ Float quadratic_target_distance(Float thickness, Float d_hat)
    {
        // Match TWP simplex projection's active boundary exactly:
        // twp_simplex_constraints uses min_distance = thickness + d_hat.
        // Keeping the Newton repulsion target identical avoids a dead band where
        // TWP projects pairs to thickness + d_hat but the recovery energy is
        // already inactive at max(thickness, d_hat).
        return thickness + d_hat;
    }

    inline __device__ Float regularized_distance(Float D, Float target)
    {
        Float eps = target * Float{1e-6};
        eps = eps > Float{1e-12} ? eps : Float{1e-12};
        return sqrt(D + eps * eps);
    }

    template <typename Grad, typename Hess>
    inline __device__ void linear_distance_derivatives(Grad&       grad_d,
                                                       Hess&       hess_d,
                                                       const Grad& grad_D,
                                                       const Hess& hess_D,
                                                       Float       distance)
    {
        Float inv_2d  = Float{0.5} / distance;
        Float inv_4d3 = Float{0.25} / (distance * distance * distance);
        grad_d = inv_2d * grad_D;
        hess_d = inv_2d * hess_D - inv_4d3 * grad_D * grad_D.transpose();
    }

    inline __device__ Float PT_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 3.0;
    }

    inline __device__ Float EE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
            }
        }
        return kappa / 4.0;
    }

    inline __device__ Float PE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector3i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 2.0;
    }

    inline __device__ Float PP_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector2i&                     cids)
    {
        ContactCoeff coeff = table(cids[0], cids[1]);
        return coeff.kappa;
    }


    inline __device__ Float PT_barrier_energy(Float          kappa,
                                              Float          d_hat,
                                              Float          thickness,
                                              const Vector3& P,
                                              const Vector3& T0,
                                              const Vector3& T1,
                                              const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ Float PT_barrier_energy(const Vector4i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  T0,
                                              const Vector3&  T1,
                                              const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ Float PT_quadratic_barrier_energy(const Vector4i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float           thickness,
                                                        const Vector3&  P,
                                                        const Vector3&  T0,
                                                        const Vector3&  T1,
                                                        const Vector3&  T2)
    {
        using namespace distance;

        Float D = 0.0;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        // Use a linear-distance gap so kappa has the same stiffness meaning as
        // PH quadratic contact. The target equals TWP's simplex min_distance.
        Float target = quadratic_target_distance(thickness, d_hat);
        Float gap = regularized_distance(D, target) - target;
        if(gap >= 0.0)
            return 0.0;

        return 0.5 * kappa * gap * gap;
    }

    inline __device__ Float PT_barrier_energy(SimplexBarrierModel model,
                                              const Vector4i&     flag,
                                              Float               kappa,
                                              Float               d_hat,
                                              Float               thickness,
                                              const Vector3&      P,
                                              const Vector3&      T0,
                                              const Vector3&      T1,
                                              const Vector3&      T2)
    {
        if(model == SimplexBarrierModel::Quadratic)
            return PT_quadratic_barrier_energy(flag, kappa, d_hat, thickness, P, T0, T1, T2);

        return PT_barrier_energy(flag, kappa, d_hat, thickness, P, T0, T1, T2);
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&      G,
                                                       Matrix12x12&   H,
                                                       Float          kappa,
                                                       Float          d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&       G,
                                                       Matrix12x12&    H,
                                                       const Vector4i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PT_quadratic_barrier_gradient_hessian(Vector12&       G,
                                                                 Matrix12x12&    H,
                                                                 const Vector4i& flag,
                                                                 Float           kappa,
                                                                 Float           d_hat,
                                                                 Float           thickness,
                                                                 const Vector3&  P,
                                                                 const Vector3&  T0,
                                                                 const Vector3&  T1,
                                                                 const Vector3&  T2)
    {
        using namespace distance;

        Float D = 0.0;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector12::Zero();
            H = Matrix12x12::Zero();
            return;
        }

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        Vector12 grad_d;
        Matrix12x12 hess_d;
        linear_distance_derivatives(grad_d, hess_d, GradD, HessD, distance);

        G = kappa * gap * grad_d;
        H = kappa * (grad_d * grad_d.transpose() + gap * hess_d);
    }

    inline __device__ void PT_barrier_gradient_hessian(SimplexBarrierModel model,
                                                       Vector12&           G,
                                                       Matrix12x12&        H,
                                                       const Vector4i&     flag,
                                                       Float               kappa,
                                                       Float               d_hat,
                                                       Float               thickness,
                                                       const Vector3&      P,
                                                       const Vector3&      T0,
                                                       const Vector3&      T1,
                                                       const Vector3&      T2)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PT_quadratic_barrier_gradient_hessian(
                G, H, flag, kappa, d_hat, thickness, P, T0, T1, T2);
            return;
        }

        PT_barrier_gradient_hessian(G, H, flag, kappa, d_hat, thickness, P, T0, T1, T2);
    }

    inline __device__ void PT_barrier_gradient(Vector12&       G,
                                               const Vector4i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  T0,
                                               const Vector3&  T1,
                                               const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PT_quadratic_barrier_gradient(Vector12&       G,
                                                         const Vector4i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  P,
                                                         const Vector3&  T0,
                                                         const Vector3&  T1,
                                                         const Vector3&  T2)
    {
        using namespace distance;

        Float D = 0.0;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector12::Zero();
            return;
        }

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);
        G = kappa * gap * (Float{0.5} / distance) * GradD;
    }

    inline __device__ void PT_barrier_gradient(SimplexBarrierModel model,
                                               Vector12&           G,
                                               const Vector4i&     flag,
                                               Float               kappa,
                                               Float               d_hat,
                                               Float               thickness,
                                               const Vector3&      P,
                                               const Vector3&      T0,
                                               const Vector3&      T1,
                                               const Vector3&      T2)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PT_quadratic_barrier_gradient(G, flag, kappa, d_hat, thickness, P, T0, T1, T2);
            return;
        }

        PT_barrier_gradient(G, flag, kappa, d_hat, thickness, P, T0, T1, T2);
    }


    inline __device__ Float mollified_EE_barrier_energy(const Vector4i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float thickness,
                                                        const Vector3& t0_Ea0,
                                                        const Vector3& t0_Ea1,
                                                        const Vector3& t0_Eb0,
                                                        const Vector3& t0_Eb1,
                                                        const Vector3& Ea0,
                                                        const Vector3& Ea1,
                                                        const Vector3& Eb0,
                                                        const Vector3& Eb1)
    {
        // using mollifier to improve the smoothness of the edge-edge barrier
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    inline __device__ Float EE_quadratic_barrier_energy(const Vector4i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float           thickness,
                                                        const Vector3&  Ea0,
                                                        const Vector3&  Ea1,
                                                        const Vector3&  Eb0,
                                                        const Vector3&  Eb1)
    {
        using namespace distance;

        Float D = 0.0;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        // Use a linear-distance gap so kappa has the same stiffness meaning as
        // PH quadratic contact. The target equals TWP's simplex min_distance.
        Float target = quadratic_target_distance(thickness, d_hat);
        Float gap = regularized_distance(D, target) - target;
        if(gap >= 0.0)
            return 0.0;

        return 0.5 * kappa * gap * gap;
    }

    inline __device__ Float EE_barrier_energy(SimplexBarrierModel model,
                                              const Vector4i&     flag,
                                              Float               kappa,
                                              Float               d_hat,
                                              Float               thickness,
                                              const Vector3&      t0_Ea0,
                                              const Vector3&      t0_Ea1,
                                              const Vector3&      t0_Eb0,
                                              const Vector3&      t0_Eb1,
                                              const Vector3&      Ea0,
                                              const Vector3&      Ea1,
                                              const Vector3&      Eb0,
                                              const Vector3&      Eb1)
    {
        if(model == SimplexBarrierModel::Quadratic)
            return EE_quadratic_barrier_energy(flag, kappa, d_hat, thickness, Ea0, Ea1, Eb0, Eb1);

        return mollified_EE_barrier_energy(flag,
                                           kappa,
                                           d_hat,
                                           thickness,
                                           t0_Ea0,
                                           t0_Ea1,
                                           t0_Eb0,
                                           t0_Eb1,
                                           Ea0,
                                           Ea1,
                                           Eb0,
                                           Eb1);
    }

    inline __device__ void mollified_EE_barrier_gradient_hessian(Vector12&    G,
                                                                 Matrix12x12& H,
                                                                 const Vector4i& flag,
                                                                 Float kappa,
                                                                 Float d_hat,
                                                                 Float thickness,
                                                                 const Vector3& t0_Ea0,
                                                                 const Vector3& t0_Ea1,
                                                                 const Vector3& t0_Eb0,
                                                                 const Vector3& t0_Eb1,
                                                                 const Vector3& Ea0,
                                                                 const Vector3& Ea1,
                                                                 const Vector3& Eb0,
                                                                 const Vector3& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        //tex: $$ \nabla D$$
        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        //tex: $$ \nabla^2 D$$
        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial B}{\partial D} $$
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial^2 B}{\partial D^2} $$
        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ \nabla B = \frac{\partial B}{\partial D} \nabla D$$
        Vector12 GradB = dBdD * GradD;

        //tex:
        //$$
        // \nabla^2 B = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T + \frac{\partial B}{\partial D} \nabla^2 D
        //$$
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        //tex: $$ \epsilon_x $$
        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        //tex: $$ e_k $$
        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        //tex: $$\nabla e_k$$
        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);


        //tex: $$ \nabla^2 e_k$$
        Matrix12x12 Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        //tex:
        //$$
        // G = \nabla e_k B + e_k \nabla B
        //$$
        G = Gradek * B + ek * GradB;

        //tex: $$ \nabla^2 e_k B + \nabla e_k \nabla B^T + \nabla B \nabla e_k^T + e_k \nabla^2 B$$
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
    }

    inline __device__ void mollified_EE_barrier_gradient(Vector12&       G,
                                                         const Vector4i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  t0_Ea0,
                                                         const Vector3&  t0_Ea1,
                                                         const Vector3&  t0_Eb0,
                                                         const Vector3&  t0_Eb1,
                                                         const Vector3&  Ea0,
                                                         const Vector3&  Ea1,
                                                         const Vector3&  Eb0,
                                                         const Vector3&  Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Vector12 GradB = dBdD * GradD;

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        G = Gradek * B + ek * GradB;
    }

    inline __device__ void EE_quadratic_barrier_gradient_hessian(Vector12&       G,
                                                                 Matrix12x12&    H,
                                                                 const Vector4i& flag,
                                                                 Float           kappa,
                                                                 Float           d_hat,
                                                                 Float           thickness,
                                                                 const Vector3&  Ea0,
                                                                 const Vector3&  Ea1,
                                                                 const Vector3&  Eb0,
                                                                 const Vector3&  Eb1)
    {
        using namespace distance;

        Float D = 0.0;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector12::Zero();
            H = Matrix12x12::Zero();
            return;
        }

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        Vector12 grad_d;
        Matrix12x12 hess_d;
        linear_distance_derivatives(grad_d, hess_d, GradD, HessD, distance);

        G = kappa * gap * grad_d;
        H = kappa * (grad_d * grad_d.transpose() + gap * hess_d);
    }

    inline __device__ void EE_barrier_gradient_hessian(SimplexBarrierModel model,
                                                       Vector12&           G,
                                                       Matrix12x12&        H,
                                                       const Vector4i&     flag,
                                                       Float               kappa,
                                                       Float               d_hat,
                                                       Float               thickness,
                                                       const Vector3&      t0_Ea0,
                                                       const Vector3&      t0_Ea1,
                                                       const Vector3&      t0_Eb0,
                                                       const Vector3&      t0_Eb1,
                                                       const Vector3&      Ea0,
                                                       const Vector3&      Ea1,
                                                       const Vector3&      Eb0,
                                                       const Vector3&      Eb1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            EE_quadratic_barrier_gradient_hessian(
                G, H, flag, kappa, d_hat, thickness, Ea0, Ea1, Eb0, Eb1);
            return;
        }

        mollified_EE_barrier_gradient_hessian(G,
                                              H,
                                              flag,
                                              kappa,
                                              d_hat,
                                              thickness,
                                              t0_Ea0,
                                              t0_Ea1,
                                              t0_Eb0,
                                              t0_Eb1,
                                              Ea0,
                                              Ea1,
                                              Eb0,
                                              Eb1);
    }

    inline __device__ void EE_quadratic_barrier_gradient(Vector12&       G,
                                                         const Vector4i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  Ea0,
                                                         const Vector3&  Ea1,
                                                         const Vector3&  Eb0,
                                                         const Vector3&  Eb1)
    {
        using namespace distance;

        Float D = 0.0;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector12::Zero();
            return;
        }

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);
        G = kappa * gap * (Float{0.5} / distance) * GradD;
    }

    inline __device__ void EE_barrier_gradient(SimplexBarrierModel model,
                                               Vector12&           G,
                                               const Vector4i&     flag,
                                               Float               kappa,
                                               Float               d_hat,
                                               Float               thickness,
                                               const Vector3&      t0_Ea0,
                                               const Vector3&      t0_Ea1,
                                               const Vector3&      t0_Eb0,
                                               const Vector3&      t0_Eb1,
                                               const Vector3&      Ea0,
                                               const Vector3&      Ea1,
                                               const Vector3&      Eb0,
                                               const Vector3&      Eb1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            EE_quadratic_barrier_gradient(G, flag, kappa, d_hat, thickness, Ea0, Ea1, Eb0, Eb1);
            return;
        }

        mollified_EE_barrier_gradient(G,
                                      flag,
                                      kappa,
                                      d_hat,
                                      thickness,
                                      t0_Ea0,
                                      t0_Ea1,
                                      t0_Eb0,
                                      t0_Eb1,
                                      Ea0,
                                      Ea1,
                                      Eb0,
                                      Eb1);
    }

    inline __device__ Float PE_barrier_energy(const Vector3i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  E0,
                                              const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ Float PE_quadratic_barrier_energy(const Vector3i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float           thickness,
                                                        const Vector3&  P,
                                                        const Vector3&  E0,
                                                        const Vector3&  E1)
    {
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float gap = regularized_distance(D, target) - target;
        if(gap >= 0.0)
            return 0.0;

        return 0.5 * kappa * gap * gap;
    }

    inline __device__ Float PE_barrier_energy(SimplexBarrierModel model,
                                              const Vector3i&     flag,
                                              Float               kappa,
                                              Float               d_hat,
                                              Float               thickness,
                                              const Vector3&      P,
                                              const Vector3&      E0,
                                              const Vector3&      E1)
    {
        if(model == SimplexBarrierModel::Quadratic)
            return PE_quadratic_barrier_energy(flag, kappa, d_hat, thickness, P, E0, E1);

        return PE_barrier_energy(flag, kappa, d_hat, thickness, P, E0, E1);
    }

    inline __device__ void PE_barrier_gradient_hessian(Vector9&        G,
                                                       Matrix9x9&      H,
                                                       const Vector3i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& E0,
                                                       const Vector3& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Matrix9x9 HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PE_quadratic_barrier_gradient_hessian(Vector9&        G,
                                                                 Matrix9x9&      H,
                                                                 const Vector3i& flag,
                                                                 Float           kappa,
                                                                 Float           d_hat,
                                                                 Float           thickness,
                                                                 const Vector3&  P,
                                                                 const Vector3&  E0,
                                                                 const Vector3&  E1)
    {
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector9::Zero();
            H = Matrix9x9::Zero();
            return;
        }

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Matrix9x9 HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        Vector9 grad_d;
        Matrix9x9 hess_d;
        linear_distance_derivatives(grad_d, hess_d, GradD, HessD, distance);

        G = kappa * gap * grad_d;
        H = kappa * (grad_d * grad_d.transpose() + gap * hess_d);
    }

    inline __device__ void PE_barrier_gradient_hessian(SimplexBarrierModel model,
                                                       Vector9&            G,
                                                       Matrix9x9&          H,
                                                       const Vector3i&     flag,
                                                       Float               kappa,
                                                       Float               d_hat,
                                                       Float               thickness,
                                                       const Vector3&      P,
                                                       const Vector3&      E0,
                                                       const Vector3&      E1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PE_quadratic_barrier_gradient_hessian(
                G, H, flag, kappa, d_hat, thickness, P, E0, E1);
            return;
        }

        PE_barrier_gradient_hessian(G, H, flag, kappa, d_hat, thickness, P, E0, E1);
    }

    inline __device__ void PE_barrier_gradient(Vector9&        G,
                                               const Vector3i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  E0,
                                               const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PE_quadratic_barrier_gradient(Vector9&        G,
                                                         const Vector3i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  P,
                                                         const Vector3&  E0,
                                                         const Vector3&  E1)
    {
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector9::Zero();
            return;
        }

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);
        G = kappa * gap * (Float{0.5} / distance) * GradD;
    }

    inline __device__ void PE_barrier_gradient(SimplexBarrierModel model,
                                               Vector9&            G,
                                               const Vector3i&     flag,
                                               Float               kappa,
                                               Float               d_hat,
                                               Float               thickness,
                                               const Vector3&      P,
                                               const Vector3&      E0,
                                               const Vector3&      E1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PE_quadratic_barrier_gradient(G, flag, kappa, d_hat, thickness, P, E0, E1);
            return;
        }

        PE_barrier_gradient(G, flag, kappa, d_hat, thickness, P, E0, E1);
    }

    inline __device__ Float PP_barrier_energy(const Vector2i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P0,
                                              const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ Float PP_quadratic_barrier_energy(const Vector2i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float           thickness,
                                                        const Vector3&  P0,
                                                        const Vector3&  P1)
    {
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float gap = regularized_distance(D, target) - target;
        if(gap >= 0.0)
            return 0.0;

        return 0.5 * kappa * gap * gap;
    }

    inline __device__ Float PP_barrier_energy(SimplexBarrierModel model,
                                              const Vector2i&     flag,
                                              Float               kappa,
                                              Float               d_hat,
                                              Float               thickness,
                                              const Vector3&      P0,
                                              const Vector3&      P1)
    {
        if(model == SimplexBarrierModel::Quadratic)
            return PP_quadratic_barrier_energy(flag, kappa, d_hat, thickness, P0, P1);

        return PP_barrier_energy(flag, kappa, d_hat, thickness, P0, P1);
    }

    inline __device__ void PP_barrier_gradient_hessian(Vector6&        G,
                                                       Matrix6x6&      H,
                                                       const Vector2i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P0,
                                                       const Vector3& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Matrix6x6 HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PP_quadratic_barrier_gradient_hessian(Vector6&        G,
                                                                 Matrix6x6&      H,
                                                                 const Vector2i& flag,
                                                                 Float           kappa,
                                                                 Float           d_hat,
                                                                 Float           thickness,
                                                                 const Vector3&  P0,
                                                                 const Vector3&  P1)
    {
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector6::Zero();
            H = Matrix6x6::Zero();
            return;
        }

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Matrix6x6 HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        Vector6 grad_d;
        Matrix6x6 hess_d;
        linear_distance_derivatives(grad_d, hess_d, GradD, HessD, distance);

        G = kappa * gap * grad_d;
        H = kappa * (grad_d * grad_d.transpose() + gap * hess_d);
    }

    inline __device__ void PP_barrier_gradient_hessian(SimplexBarrierModel model,
                                                       Vector6&            G,
                                                       Matrix6x6&          H,
                                                       const Vector2i&     flag,
                                                       Float               kappa,
                                                       Float               d_hat,
                                                       Float               thickness,
                                                       const Vector3&      P0,
                                                       const Vector3&      P1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PP_quadratic_barrier_gradient_hessian(G, H, flag, kappa, d_hat, thickness, P0, P1);
            return;
        }

        PP_barrier_gradient_hessian(G, H, flag, kappa, d_hat, thickness, P0, P1);
    }

    inline __device__ void PP_barrier_gradient(Vector6&        G,
                                               const Vector2i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P0,
                                               const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ void PP_quadratic_barrier_gradient(Vector6&        G,
                                                         const Vector2i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float           thickness,
                                                         const Vector3&  P0,
                                                         const Vector3&  P1)
    {
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Float target = quadratic_target_distance(thickness, d_hat);
        Float distance = regularized_distance(D, target);
        Float gap = distance - target;
        if(gap >= 0.0)
        {
            G = Vector6::Zero();
            return;
        }

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);
        G = kappa * gap * (Float{0.5} / distance) * GradD;
    }

    inline __device__ void PP_barrier_gradient(SimplexBarrierModel model,
                                               Vector6&            G,
                                               const Vector2i&     flag,
                                               Float               kappa,
                                               Float               d_hat,
                                               Float               thickness,
                                               const Vector3&      P0,
                                               const Vector3&      P1)
    {
        if(model == SimplexBarrierModel::Quadratic)
        {
            PP_quadratic_barrier_gradient(G, flag, kappa, d_hat, thickness, P0, P1);
            return;
        }

        PP_barrier_gradient(G, flag, kappa, d_hat, thickness, P0, P1);
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda
