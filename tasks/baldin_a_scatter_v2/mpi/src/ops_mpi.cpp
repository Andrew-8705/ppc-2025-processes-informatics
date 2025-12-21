#include "baldin_a_scatter_v2/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "baldin_a_scatter_v2/common/include/common.hpp"

namespace baldin_a_scatter_v2 {

BaldinAScatterV2MPI::BaldinAScatterV2MPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAScatterV2MPI::ValidationImpl() {
  const auto &args = GetInput();

  bool is_invalid_counts = (args.send_count <= 0) || (args.send_count != args.recv_count);
  bool is_invalid_root = (args.root_rank < 0);

  if (is_invalid_counts || is_invalid_root) {
    return false;
  }

  if (args.send_type != args.recv_type) {
    return false;
  }

  auto is_supported = [](MPI_Datatype t) { return (t == MPI_INT || t == MPI_FLOAT || t == MPI_DOUBLE); };

  return is_supported(args.send_type);
}

bool BaldinAScatterV2MPI::PreProcessingImpl() {
  auto &args = GetInput();
  int comm_size = 0;
  MPI_Comm_size(args.comm, &comm_size);

  if (args.root_rank >= comm_size) {
    args.root_rank %= comm_size;
  }

  return true;
}

namespace {
size_t get_type_byte_size(MPI_Datatype type) {
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);
  return static_cast<size_t>(extent);
}

int map_virt_to_real(int virt_rank, int root, int size) {
  return (virt_rank + root) % size;
}

std::vector<uint8_t> prepare_root_data(const void *src, int size, int root, int count, size_t type_size) {
  size_t chunk_bytes = static_cast<size_t>(count) * type_size;
  size_t total_bytes = static_cast<size_t>(size) * chunk_bytes;

  std::vector<uint8_t> buffer(total_bytes);

  const uint8_t *src_bytes = static_cast<const uint8_t *>(src);
  uint8_t *dst_bytes = buffer.data();

  size_t offset = root * chunk_bytes;
  size_t tail_size = total_bytes - offset;

  std::copy(src_bytes + offset, src_bytes + total_bytes, dst_bytes);

  std::copy(src_bytes, src_bytes + offset, dst_bytes + tail_size);

  return buffer;
}

void exec_scatter_cycle(int size, int root, int rank, int count, MPI_Datatype type, size_t type_size, MPI_Comm comm,
                        const uint8_t *&active_ptr, std::vector<uint8_t> &buffer) {
  int relative_rank = (rank - root + size) % size;

  int start_stride = 1;
  while (start_stride < size) {
    start_stride <<= 1;
  }
  start_stride >>= 1;

  for (int stride = start_stride; stride > 0; stride >>= 1) {
    if (relative_rank % stride != 0) {
      continue;
    }

    bool is_sender = (relative_rank % (stride << 1) == 0);

    if (is_sender) {
      int virt_dest = relative_rank + stride;

      if (virt_dest < size) {
        int limit = std::min(virt_dest + stride, size);
        int send_amt = (limit - virt_dest) * count;

        size_t byte_shift = static_cast<size_t>(virt_dest - relative_rank) * count * type_size;
        int real_dest = map_virt_to_real(virt_dest, root, size);

        MPI_Send(active_ptr + byte_shift, send_amt, type, real_dest, 0, comm);
      }
    } else {
      int virt_src = relative_rank - stride;
      int real_src = map_virt_to_real(virt_src, root, size);

      int limit = std::min(relative_rank + stride, size);
      int recv_amt = (limit - relative_rank) * count;

      size_t required_bytes = static_cast<size_t>(recv_amt) * type_size;
      buffer.resize(required_bytes);

      MPI_Recv(buffer.data(), recv_amt, type, real_src, 0, comm, MPI_STATUS_IGNORE);

      active_ptr = buffer.data();
    }
  }
}

}  // namespace

bool BaldinAScatterV2MPI::RunImpl() {
  auto &args = GetInput();

  int rank, size;
  MPI_Comm_rank(args.comm, &rank);
  MPI_Comm_size(args.comm, &size);

  size_t type_size = get_type_byte_size(args.recv_type);

  std::vector<uint8_t> internal_buf;
  const uint8_t *active_data_ptr = nullptr;

  if (rank == args.root_rank) {
    internal_buf = prepare_root_data(args.src_buffer, size, args.root_rank, args.recv_count, type_size);
    active_data_ptr = internal_buf.data();
  }

  exec_scatter_cycle(size, args.root_rank, rank, args.recv_count, args.recv_type, type_size, args.comm, active_data_ptr,
                     internal_buf);

  if (args.dst_buffer != MPI_IN_PLACE && active_data_ptr != nullptr) {
    size_t bytes_to_copy = args.recv_count * type_size;
    uint8_t *user_dst = static_cast<uint8_t *>(args.dst_buffer);
    std::copy(active_data_ptr, active_data_ptr + bytes_to_copy, user_dst);
  }

  size_t result_bytes = args.recv_count * type_size;
  std::vector<uint8_t> result(result_bytes);

  const void *final_src = (args.dst_buffer != nullptr) ? args.dst_buffer : active_data_ptr;
  if (final_src) {
    const uint8_t *ptr = static_cast<const uint8_t *>(final_src);
    std::copy(ptr, ptr + result_bytes, result.begin());
  }

  GetOutput() = std::move(result);
  return true;
}

bool BaldinAScatterV2MPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_scatter_v2
