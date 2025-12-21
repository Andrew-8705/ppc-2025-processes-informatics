#pragma once

#include "baldin_a_scatter_v2/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_scatter_v2 {

class BaldinAScatterV2MPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaldinAScatterV2MPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_scatter_v2
