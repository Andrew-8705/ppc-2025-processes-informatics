#include "baldin_a_scatter_v2/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "baldin_a_scatter_v2/common/include/common.hpp"

namespace baldin_a_scatter_v2 {

BaldinAScatterV2MPI::BaldinAScatterV2MPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAScatterV2MPI::ValidationImpl() {
  const auto &input = GetInput();
  int sendcount = input.send_count;
  int recvcount = input.recv_count;
  int root = input.root_rank;
  MPI_Datatype sendtype = input.send_type;
  MPI_Datatype recvtype = input.recv_type;

  if (sendcount <= 0 || sendcount != recvcount || root < 0) {
    return false;
  }
  if (sendtype != recvtype) {
    return false;
  }

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return is_sup_type(sendtype);
}

bool BaldinAScatterV2MPI::PreProcessingImpl() {
  auto &input = GetInput();
  int root = input.root_rank;

  int world_size = 0;
  MPI_Comm_size(input.comm, &world_size);

  if (root >= world_size) {
    input.root_rank = root % world_size;
  }

  return true;
}

namespace {

MPI_Aint GetDataTypeExtent(MPI_Datatype type) {
  MPI_Aint lb = 0;
  MPI_Aint extent = 0;
  MPI_Type_get_extent(type, &lb, &extent);
  return extent;
}

int VirtualToRealRank(int v_rank, int root, int size) {
  return (v_rank + root) % size;
}

int CalculateSubtreeSize(int v_dest, int mask, int size) {
  return std::min(v_dest + mask, size);
}

int CalculateInitialMask(int size) {
  int mask = 1;
  while (mask < size) {
    mask <<= 1;
  }
  return mask >> 1;
}

void PrepareRootBuffer(const void *sendbuf, int size, int root, int count, MPI_Aint extent, std::vector<char> &buffer) {
  size_t total_bytes = static_cast<size_t>(size) * count * extent;
  size_t chunk_bytes = static_cast<size_t>(count) * extent;

  buffer.resize(total_bytes);

  const char *send_ptr = static_cast<const char *>(sendbuf);
  char *tmp_ptr = buffer.data();

  size_t first_part_bytes = (size - root) * chunk_bytes;
  size_t second_part_bytes = root * chunk_bytes;

  std::memcpy(tmp_ptr, send_ptr + second_part_bytes, first_part_bytes);
  std::memcpy(tmp_ptr + first_part_bytes, send_ptr, second_part_bytes);
}

}  // namespace

bool BaldinAScatterV2MPI::RunImpl() {
  auto &input = GetInput();

  const void *sendbuf = input.src_buffer;
  int sendcount = input.send_count;
  MPI_Datatype sendtype = input.send_type;
  void *recvbuf = input.dst_buffer;
  int recvcount = input.recv_count;
  MPI_Datatype recvtype = input.recv_type;
  int root = input.root_rank;
  MPI_Comm comm = input.comm;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Aint extent = GetDataTypeExtent(rank == root ? sendtype : recvtype);

  std::vector<char> temp_buffer;
  const char *curr_buf_ptr = nullptr;

  if (rank == root) {
    PrepareRootBuffer(sendbuf, size, root, sendcount, extent, temp_buffer);
    curr_buf_ptr = temp_buffer.data();
  }

  int v_rank = (rank - root + size) % size;
  int mask = CalculateInitialMask(size);

  while (mask > 0) {
    if (v_rank % (2 * mask) == 0) {
      int v_dest = v_rank + mask;

      if (v_dest < size) {
        int subtree_size = CalculateSubtreeSize(v_dest, mask, size);
        int count_to_send = (subtree_size - v_dest) * recvcount;

        size_t offset_bytes = static_cast<size_t>(v_dest - v_rank) * recvcount * extent;
        int real_dest = VirtualToRealRank(v_dest, root, size);

        MPI_Send(curr_buf_ptr + offset_bytes, count_to_send, (rank == root ? sendtype : recvtype), real_dest, 0, comm);
      }
    }

    else if (v_rank % (2 * mask) == mask) {
      int v_source = v_rank - mask;
      int real_source = VirtualToRealRank(v_source, root, size);

      int subtree_end = CalculateSubtreeSize(v_rank, mask, size);
      int count_to_recv = (subtree_end - v_rank) * recvcount;

      size_t bytes_to_recv = static_cast<size_t>(count_to_recv) * extent;
      temp_buffer.resize(bytes_to_recv);

      MPI_Recv(temp_buffer.data(), count_to_recv, recvtype, real_source, 0, comm, MPI_STATUS_IGNORE);

      curr_buf_ptr = temp_buffer.data();
    }

    mask >>= 1;
  }

  if (recvbuf != MPI_IN_PLACE && curr_buf_ptr != nullptr) {
    std::memcpy(recvbuf, curr_buf_ptr, recvcount * extent);
  }

  size_t out_bytes = static_cast<size_t>(recvcount) * extent;
  std::vector<uint8_t> output_vec(out_bytes);
  if (recvbuf != nullptr) {
    std::memcpy(output_vec.data(), recvbuf, out_bytes);
  }

  GetOutput() = std::move(output_vec);
  return true;
}

bool BaldinAScatterV2MPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_scatter_v2
