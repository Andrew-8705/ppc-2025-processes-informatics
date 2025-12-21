#pragma once

#include <mpi.h>

#include <tuple>

#include "task/include/task.hpp"

namespace baldin_a_scatter_v2 {

struct ScatterArgs {
  const void *src_buffer;
  int send_count;
  MPI_Datatype send_type;

  void *dst_buffer;
  int recv_count;
  MPI_Datatype recv_type;

  int root_rank;
  MPI_Comm comm;
};

using InType = ScatterArgs;
using OutType = std::vector<uint8_t>;
using TestType = ScatterArgs;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_scatter_v2
