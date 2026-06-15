#pragma once
#include <sim_system.h>
#include <string_view>

namespace uipc::backend::cuda
{
class GlobalTWP final : public SimSystem
{
  public:
    explicit GlobalTWP(SimEngine& engine);
    ~GlobalTWP() override;

    class Impl;

  private:
    friend class SimEngine;

    void do_build() override;
    void init();
    void project();
    void debug_log_state(std::string_view stage);

    U<Impl> m_impl;
};
}  // namespace uipc::backend::cuda
