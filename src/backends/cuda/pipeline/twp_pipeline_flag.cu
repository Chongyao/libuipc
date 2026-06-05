#include <pipeline/twp_pipeline_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda::TWPPipelineFlag>
{
  public:
    static U<cuda::TWPPipelineFlag> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda::SimEngine&>(engine).world().scene();
        auto ctype_attr = scene.config().find<std::string>("contact/constitution");

        if(ctype_attr->view()[0] != "twp")
        {
            return nullptr;
        }
        return uipc::make_unique<cuda::TWPPipelineFlag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(TWPPipelineFlag);
void TWPPipelineFlag::do_build() {}
}  // namespace uipc::backend::cuda
