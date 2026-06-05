#pragma once
#include <sim_system.h>

namespace uipc::backend::cuda
{
/*
* @brief TWP pipeline flag
*
* The first stage reuses IPC contact detection systems, but keeps contact out of
* the Newton solve and calls a projection pass at the end of the timestep.
*/
class TWPPipelineFlag final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    void do_build() override;
};
}  // namespace uipc::backend::cuda
