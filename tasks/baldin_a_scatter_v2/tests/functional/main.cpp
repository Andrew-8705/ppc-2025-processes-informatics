#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "baldin_a_scatter_v2/common/include/common.hpp"
#include "baldin_a_scatter_v2/mpi/include/ops_mpi.hpp"
#include "baldin_a_scatter_v2/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_scatter_v2 {

namespace {
std::string GetMpiTypeName(MPI_Datatype t) {
  if (t == MPI_INT) {
    return "INT";
  }
  if (t == MPI_FLOAT) {
    return "FLOAT";
  }
  if (t == MPI_DOUBLE) {
    return "DOUBLE";
  }
  return "UNKNOWN";
}
}  // namespace

class BaldinAScatterV2FuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &p) {
    return GetMpiTypeName(p.send_type) + "_C" + std::to_string(p.send_count) + "_R" + std::to_string(p.root_rank);
  }

  bool static IsModeSeq() {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    return (std::string(info->name()).find("seq") != std::string::npos);
  }

 protected:
  std::vector<int> data_i;
  std::vector<float> data_f;
  std::vector<double> data_d;

  std::vector<uint8_t> recv_buf_bytes;
  InType task_args;

  void SetUp() override {
    TestType p = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int count = p.send_count;
    int root = p.root_rank;
    MPI_Datatype type = p.send_type;

    bool is_seq = IsModeSeq();

    int my_rank = 0;
    int total_p = 1;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &total_p);
    }

    int real_root = root % total_p;
    bool is_root = (my_rank == real_root);

    size_t sz = 0;
    if (type == MPI_INT) {
      sz = sizeof(int);
    } else if (type == MPI_FLOAT) {
      sz = sizeof(float);
    } else if (type == MPI_DOUBLE) {
      sz = sizeof(double);
    } else {
      return;
    }

    recv_buf_bytes.resize(count * sz);

    const void *src_ptr = nullptr;
    if (is_root) {
      size_t total_elems = is_seq ? count : (count * total_p);

      if (type == MPI_INT) {
        data_i.resize(total_elems);
        for (size_t k = 0; k < total_elems; ++k) {
          data_i[k] = static_cast<int>(k * 3 + 7);
        }
        src_ptr = data_i.data();
      } else if (type == MPI_FLOAT) {
        data_f.resize(total_elems);
        for (size_t k = 0; k < total_elems; ++k) {
          data_f[k] = static_cast<float>(k * 1.5f - 2.2f);
        }
        src_ptr = data_f.data();
      } else if (type == MPI_DOUBLE) {
        data_d.resize(total_elems);
        for (size_t k = 0; k < total_elems; ++k) {
          data_d[k] = static_cast<double>(k * 0.12345 + 9.8765);
        }
        src_ptr = data_d.data();
      }
    }

    task_args.src_buffer = src_ptr;
    task_args.send_count = count;
    task_args.send_type = type;
    task_args.dst_buffer = recv_buf_bytes.data();
    task_args.recv_count = count;
    task_args.recv_type = type;
    task_args.root_rank = root;
    task_args.comm = MPI_COMM_WORLD;
  }

  bool CheckTestOutputData(OutType &out) final {
    TestType p = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int cnt = p.send_count;
    MPI_Datatype t = p.send_type;

    if (out.empty()) {
      return false;
    }

    bool is_seq = IsModeSeq();
    int r = 0;
    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
    }

    size_t start_idx = static_cast<size_t>(r) * cnt;
    const void *ptr = out.data();

    if (t == MPI_INT) {
      if (out.size() != cnt * sizeof(int)) {
        return false;
      }
      const int *arr = reinterpret_cast<const int *>(ptr);
      for (int k = 0; k < cnt; ++k) {
        if (arr[k] != static_cast<int>((start_idx + k) * 3 + 7)) {
          return false;
        }
      }
    } else if (t == MPI_FLOAT) {
      if (out.size() != cnt * sizeof(float)) {
        return false;
      }
      const float *arr = reinterpret_cast<const float *>(ptr);
      for (int k = 0; k < cnt; ++k) {
        float expected = static_cast<float>((start_idx + k) * 1.5f - 2.2f);
        if (std::abs(arr[k] - expected) >= 1e-5) {
          return false;
        }
      }
    } else if (t == MPI_DOUBLE) {
      if (out.size() != cnt * sizeof(double)) {
        return false;
      }
      const double *arr = reinterpret_cast<const double *>(ptr);
      for (int k = 0; k < cnt; ++k) {
        double expected = static_cast<double>((start_idx + k) * 0.12345 + 9.8765);
        if (std::abs(arr[k] - expected) >= 1e-9) {
          return false;
        }
      }
    } else {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return task_args;
  }
};

namespace {

TEST_P(BaldinAScatterV2FuncTests, ScatterTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParams = {
    ScatterArgs{nullptr, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 5, MPI_FLOAT, nullptr, 5, MPI_FLOAT, 0, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 3, MPI_DOUBLE, nullptr, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD},

    ScatterArgs{nullptr, 10, MPI_INT, nullptr, 10, MPI_INT, 1, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 10, MPI_FLOAT, nullptr, 10, MPI_FLOAT, 2, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 10, MPI_DOUBLE, nullptr, 10, MPI_DOUBLE, 3, MPI_COMM_WORLD},

    ScatterArgs{nullptr, 20, MPI_INT, nullptr, 20, MPI_INT, 0, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 15, MPI_FLOAT, nullptr, 15, MPI_FLOAT, 1, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 25, MPI_DOUBLE, nullptr, 25, MPI_DOUBLE, 2, MPI_COMM_WORLD},

    ScatterArgs{nullptr, 1000, MPI_INT, nullptr, 1000, MPI_INT, 0, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 1000, MPI_FLOAT, nullptr, 1000, MPI_FLOAT, 1, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 1000, MPI_DOUBLE, nullptr, 1000, MPI_DOUBLE, 2, MPI_COMM_WORLD},

    ScatterArgs{nullptr, 7, MPI_INT, nullptr, 7, MPI_INT, 1, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 13, MPI_FLOAT, nullptr, 13, MPI_FLOAT, 3, MPI_COMM_WORLD},
    ScatterArgs{nullptr, 21, MPI_DOUBLE, nullptr, 21, MPI_DOUBLE, 0, MPI_COMM_WORLD}};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAScatterV2MPI, InType>(kTestParams, PPC_SETTINGS_baldin_a_scatter_v2),
                   ppc::util::AddFuncTask<BaldinAScatterV2SEQ, InType>(kTestParams, PPC_SETTINGS_baldin_a_scatter_v2));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAScatterV2FuncTests::PrintFuncTestName<BaldinAScatterV2FuncTests>;

INSTANTIATE_TEST_SUITE_P(ScatterFuncTests, BaldinAScatterV2FuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace baldin_a_scatter_v2
