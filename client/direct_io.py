#!/usr/bin/env python3

# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ctypes

def print_memoryview(mv):
    """Print the address and size of a memoryview."""
    address = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    size = mv.nbytes
    print(f"Address: {hex(address)}, Size: {hex(size)}")

def allocate_aligned_buffers(buffer_size, alignment, num_bufs):
    """Allocate a sequence of aligned buffers.

    Args:
        buffer_size (int): Size of each buffer in bytes.
        alignment (int): Alignment boundary in bytes.
        num_bufs (int): Number of buffers to allocate.

    Returns:
        list[memoryview]: A list of memoryview objects, each representing an aligned buffer
        raw_buf: the ctypes raw string_buffer.  This *must* be kept to avoid the
        gc prematurely collecting the buffers.
    """
    assert buffer_size % alignment == 0, "buffer_size must be a multiple of alignment"

    # Allocate a single large buffer that can accommodate all buffers with proper alignment
    total_size = (num_bufs * buffer_size) + alignment
    raw_buf = ctypes.create_string_buffer(total_size)
    base_address = ctypes.addressof(raw_buf)

    # Calculate the offset to align the first buffer
    first_offset = (alignment - (base_address % alignment)) % alignment
    aligned_base_address = base_address + first_offset

    # Create memoryviews for each buffer
    buffers = []
    for i in range(num_bufs):
        buf_address = aligned_base_address + i * buffer_size
        aligned_buf = (ctypes.c_char * buffer_size).from_address(buf_address)
        mv = memoryview(aligned_buf)
        # print_memoryview(mv)
        buffers.append(mv)
    return buffers, raw_buf

def round_up(value, multiple):
    """Round up `value` to the next multiple of `multiple`."""
    if multiple == 0:
        raise ValueError("Multiple must be greater than 0.")
    return ((value + multiple - 1) // multiple) * multiple

def pread(fd, aligned_memview, count, offset, fs_block_size, thread_id):
    """Perform a direct pread.

    Returns:
        number of bytes read. Unlike Posix pread this routine handles
        interrupts, so return value should always be equal to count unless
        there is an unrecoverable error, in which case it throws rather than returns

    Required: `count` <= aligned_memview size
    """
    padded_count = round_up(count, fs_block_size)
    assert padded_count <= aligned_memview.nbytes, "memview too small for requested read"

    while True:
        try:
            bytes_read = os.preadv(fd, [aligned_memview[:padded_count]], offset)

            if bytes_read == count:
                # Expected case: all requested bytes were read.
                return bytes_read

            if bytes_read > 0:
                # Retry case: partial read, retry the entire read to maintain alignment restrictions
                assert bytes_read < count, "bytes_read cannot exceed count"
                continue

            if bytes_read == 0:
                # Unexpected EOF
                raise RuntimeError(f"Unexpected EOF encountered at offset {offset}")

            if bytes_read < 0:
                raise OSError("preadv returned a negative value")

        except InterruptedError:
            # Retry on EINTR
            continue
        except OSError as e:
            # Non-retriable errors
            if e.errno == 14:  # Errno 14 corresponds to "Bad address"
                raise RuntimeError(f"preadv failed with error: {e}, this usually means out of memory")
            raise RuntimeError(f"preadv failed with error: {e}")

def pwrite(fd, aligned_memview, count, offset, fs_block_size):
    """Perform a direct pwrite

    Returns:
        number of bytes written. Unlike Posix pwrite this routine handles
        interrupts, so return value should always be equal to count unless
        there is an unrecoverable error, in which case it throws rather than
        returns

    Required: `count <= aligned_memview size
    """
    padded_count = round_up(count, fs_block_size)
    assert padded_count <= aligned_memview.nbytes, "memview too small for requested write"
    while True:
        try:
            bytes_written = os.pwritev(fd, [aligned_memview[:padded_count]], offset)

            if bytes_written == padded_count:
                # Expected case: all requested bytes were written.
                # in the case of the last block in the file we may have padded up, so return
                # the bytes _requested_ rather than the padded count
                return count

            if bytes_written > 0:
                # Retry case: partial write, retry the entire write to maintain alignment restrictions
                assert bytes_written < count, "bytes_written cannot exceed count"
                continue

            if bytes_written == 0:
                # Unexpected failure to write
                raise RuntimeError(f"Unexpected failure to write at offset {offset}")

            if bytes_written < 0:
                raise OSError("pwritev returned a negative value")

        except InterruptedError:
            # Retry on EINTR
            continue
        except OSError as e:
            # Non-retriable errors
            if e.errno == 14:  # Errno 14 corresponds to "Bad address"
                raise RuntimeError(f"pwritev failed with error: {e}, this usually means out of memory")
            raise RuntimeError(f"pwritev failed with error: {e}")
