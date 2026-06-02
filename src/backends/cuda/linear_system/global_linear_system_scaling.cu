#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
namespace
{
__device__ Matrix3x3 block_cholesky_inverse_lower(const Matrix3x3& A, Float eps)
{
    Matrix3x3 S = Matrix3x3::Identity();

    const Float a00 = A(0, 0) + eps;
    if(!(a00 > eps))
        return S;
    const Float l00 = sqrt(a00);

    const Float l10 = A(1, 0) / l00;
    const Float l20 = A(2, 0) / l00;

    const Float d11 = A(1, 1) + eps - l10 * l10;
    if(!(d11 > eps))
        return S;
    const Float l11 = sqrt(d11);

    const Float l21 = (A(2, 1) - l20 * l10) / l11;

    const Float d22 = A(2, 2) + eps - l20 * l20 - l21 * l21;
    if(!(d22 > eps))
        return S;
    const Float l22 = sqrt(d22);

    S.setZero();
    S(0, 0) = 1.0 / l00;
    S(1, 0) = -l10 / (l00 * l11);
    S(1, 1) = 1.0 / l11;
    S(2, 0) = (l10 * l21 - l20 * l11) / (l00 * l11 * l22);
    S(2, 1) = -l21 / (l11 * l22);
    S(2, 2) = 1.0 / l22;

    return S;
}
}  // namespace

void GlobalLinearSystem::Impl::apply_block_diagonal_scaling()
{
    if(!block_diagonal_scaling_enabled)
        return;

    Timer timer{"Block Diagonal Scaling"};

    using namespace muda;
    constexpr int N = 3;

    auto block_count = b.size() / N;
    block_diag_scalings.resize(block_count);
    block_diag_scalings.fill(Matrix3x3::Identity());

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(bcoo_A.triplet_count(),
               [A       = bcoo_A.cview().cviewer().name("A"),
                scalings = block_diag_scalings.viewer().name("block_diag_scalings"),
                eps      = block_diagonal_scaling_eps] __device__(int k) mutable
               {
                   auto&& [i, j, block] = A(k);
                   if(i == j)
                       scalings(i) = block_cholesky_inverse_lower(block, eps);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(bcoo_A.triplet_count(),
               [rows     = bcoo_A.row_indices().cviewer().name("rows"),
                cols     = bcoo_A.col_indices().cviewer().name("cols"),
                values   = bcoo_A.values().viewer().name("values"),
                scalings = block_diag_scalings.cviewer().name("block_diag_scalings")]
               __device__(int k) mutable
               {
                   const int i = rows(k);
                   const int j = cols(k);
                   values(k) = scalings(i) * values(k) * scalings(j).transpose();
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(block_count,
               [rhs      = b.view().viewer().name("b"),
                scalings = block_diag_scalings.cviewer().name("block_diag_scalings")]
               __device__(int i) mutable
               {
                   rhs.segment<N>(i * N).as_eigen() =
                       scalings(i) * rhs.segment<N>(i * N).as_eigen();
               });
}

void GlobalLinearSystem::Impl::unscale_solution()
{
    if(!block_diagonal_scaling_enabled)
        return;

    using namespace muda;
    constexpr int N = 3;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(block_diag_scalings.size(),
               [solution = x.view().viewer().name("x"),
                scalings = block_diag_scalings.cviewer().name("block_diag_scalings")]
               __device__(int i) mutable
               {
                   solution.segment<N>(i * N).as_eigen() =
                       scalings(i).transpose() * solution.segment<N>(i * N).as_eigen();
               });
}
}  // namespace uipc::backend::cuda
