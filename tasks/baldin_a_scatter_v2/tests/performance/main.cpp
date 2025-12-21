#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "baldin_a_scatter_v2/common/include/common.hpp"
#include "baldin_a_scatter_v2/mpi/include/ops_mpi.hpp"
#include "baldin_a_scatter_v2/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baldin_a_scatter_v2 {

class BaldinAScatterV2PerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<int> send_vec_;
  std::vector<int> recv_vec_;

  const int count_per_proc_ = 1000000;

  bool static IsModeSeq() {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    return (std::string(info->name()).find("seq") != std::string::npos);
  }

  void SetUp() override {
    bool is_seq = IsModeSeq();

    int rank = 0;
    int size = 1;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int root = 0;

    recv_vec_.resize(count_per_proc_);
    bool is_root = is_seq || (rank == root);

    if (is_root) {
      size_t total_send_count = is_seq ? count_per_proc_ : (count_per_proc_ * size);

      send_vec_.resize(total_send_count);
      // Заполняем данными: i -> i * 5 - 13
      for (size_t i = 0; i < total_send_count; i++) {
        send_vec_[i] = static_cast<int>(i * 5 - 13);
      }
    }

    const void *sendbuf_ptr = is_root ? send_vec_.data() : nullptr;

    input_data_.src_buffer = sendbuf_ptr;
    input_data_.send_count = count_per_proc_;
    input_data_.send_type = MPI_INT;
    input_data_.dst_buffer = recv_vec_.data();
    input_data_.recv_count = count_per_proc_;
    input_data_.recv_type = MPI_INT;
    input_data_.root_rank = root;
    input_data_.comm = MPI_COMM_WORLD;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    bool is_seq = IsModeSeq();
    int rank = 0;
    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (input_data_.send_type == MPI_INT) {
      long long base_global_index = static_cast<long long>(rank) * count_per_proc_;

      const int *actual_data = reinterpret_cast<const int *>(output_data.data());

      if (output_data.size() != count_per_proc_ * sizeof(int)) {
        return false;
      }

      for (int i = 0; i < count_per_proc_; ++i) {
        int expected_val = static_cast<int>((base_global_index + i) * 5 - 13);
        if (actual_data[i] != expected_val) {
          return false;
        }
      }
    } else {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BaldinAScatterV2PerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BaldinAScatterV2MPI, BaldinAScatterV2SEQ>(PPC_SETTINGS_baldin_a_scatter_v2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaldinAScatterV2PerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAScatterV2PerfTests, kGtestValues, kPerfTestName);

}  // namespace baldin_a_scatter_v2
