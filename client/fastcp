#!/usr/bin/env python3

import os
import argparse
import ctypes
from ctypes import c_char, c_size_t, POINTER
import tempfile
import ctypes
from queue import Queue
import threading

def print_memoryview(mv):
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
        list[memoryview]: A list of memoryview objects, each representing an
        aligned buffer

        raw_buf the ctypes raw string_buffer.  This *must* be kept to avoid the
        gc prematurely collecting the buffers.

    """
    # Ensure buffer_size is a multiple of alignment
    assert buffer_size % alignment == 0, "buffer_size must be a multiple of alignment"
    
    # Allocate a single large buffer that can accommodate all buffers with proper alignment
    total_size = (num_bufs * buffer_size) + alignment
    raw_buf = ctypes.create_string_buffer(total_size)
    base_address = ctypes.addressof(raw_buf)
    
    # Calculate the offset to align the first buffer
    first_offset = (alignment - (base_address % alignment)) % alignment
    aligned_base_address = base_address + first_offset
    # print(f"allocate_aligned_buffers: base_address {hex(base_address)}, aligned {hex(aligned_base_address)}")
    
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

def pread_direct(fd, aligned_memview, count, offset, fs_block_size, thread_id):
    """Perform a direct pread.

    Returns:
        number of bytes read. Unlike Posix pread this routine handles
        interrupts, so return value should always be equal to count unless
        there is an unrecoverable error, in which case it throws rather than returns

    Required: `count` <= aligned_memview size

    """

    padded_count = round_up(count, fs_block_size)  # Ensure count is a multiple of fs_block_size
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

def pwrite_direct(fd, aligned_memview, count, offset, fs_block_size):
    """Perform a direct pwrite

    Returns:
        number of bytes written. Unlike Posix pwrite this routine handles
        interrupts, so return value should always be equal to count unless
        there is an unrecoverable error, in which case it throws rather than
        returns

    Required: `count <= aligned_memview size

    """
    padded_count = round_up(count, fs_block_size)  # Ensure count is a multiple of fs_block_size
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

def copy_worker(fd_src, fd_dst, buffer_size, workpile, thread_buffer, fs_block_size, thread_id):
    """Worker function to copy chunks from the workpile."""
    while True:
        try:
            # Get the next chunk to copy.
            offset, size = workpile.get_nowait()
        except Exception:
            # Workpile is empty, so exit.
            return

        # pread_direct and pwrite_direct have their own retry loop so we don't need one here
        bytes_read = pread_direct(fd_src, thread_buffer, size, offset, fs_block_size, thread_id)
        assert bytes_read == size, "pread returned different size than requested"
        bytes_written = pwrite_direct(fd_dst, thread_buffer, bytes_read, offset, fs_block_size)
        assert bytes_written == size, "pwrite returned different size than requested"
        workpile.task_done()
            

def fastcp(src_path, dst_path, num_threads, buffer_size):
    """Copy a file using multiple threads with a workpile algorithm."""

    fd_src, fd_dst = None, None
    print ("opening direct")
    try:
        # Open files as file descriptors with O_DIRECT.
        try:
            fd_src = os.open(src_path, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            raise RuntimeError(f"Failed to open source file '{src_path}': {e}")

        try:
            fd_dst = os.open(dst_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
        except OSError as e:
            raise RuntimeError(f"Failed to open destination file '{dst_path}': {e}")

        try:
            # Get the size of the source file.
            file_size = os.fstat(fd_src).st_size
            print("file size is", file_size)

            # Get file system block sizes.
            fs_block_size = os.fstatvfs(fd_src).f_bsize
            dst_block_size = os.fstatvfs(fd_dst).f_bsize

            # get each thread an aligned buffer
            
            alignment = 2*1024*1024 # 2M is the size of Linux Huge Pages, so this should always be enough
            assert alignment >= max(fs_block_size, dst_block_size, 512), "alignment smaller than fs block size"

            thread_buffers, raw_buf = allocate_aligned_buffers(buffer_size, alignment, num_threads)

            # Create the workpile with all file chunks.
            chunk_size = buffer_size
            workpile = Queue()
            for offset in range(0, file_size, chunk_size):
                workpile.put((offset, min(chunk_size, file_size - offset)))

            # Start worker threads.
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=copy_worker,
                                          args=(fd_src, fd_dst, buffer_size, workpile,
                                                thread_buffers[i], fs_block_size, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish.
            for thread in threads:
                thread.join()

            # Set the size of the destination file.
            os.ftruncate(fd_dst, file_size)
        except Exception as e:
            raise RuntimeError(f"Error during file copy operation: {e}")
    finally:
        if fd_src is not None:
            os.close(fd_src)
        if fd_dst is not None:
            os.close(fd_dst)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Threaded file copy utility.")
    parser.add_argument("src", help="Source file path")
    parser.add_argument("dst", help="Destination file path")
    parser.add_argument("-n", "--num-threads", type=int, default=64, help="Number of threads (default: 4)")
    parser.add_argument("-b", "--buffer-size", type=int, default=256 * 1024 * 1024, help="Buffer size in bytes (default: 256MB, must be a multiple of 2MiB)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Check that buffer_size is a multiple of 2 MiB
    if args.buffer_size % (2 * 1024 * 1024) != 0:
        raise ValueError("Buffer size must be a multiple of 2 MiB.")

    fastcp(args.src, args.dst, args.num_threads, args.buffer_size)

if __name__ == "__main__":
    main()
    exit(0)
    
    # Run a test harness when no arguments are provided
    with tempfile.NamedTemporaryFile(delete=False) as src_file:
        src_file.write(os.urandom(16000123))  # Generate a random source file
#        src_file.write(os.urandom(2*1024*1024))  # Generate a random source file
        src_path = src_file.name

    with tempfile.NamedTemporaryFile(delete=False) as dst_file:
        dst_path = dst_file.name

    try:
        print(f"Running test harness with source: {src_path}, destination: {dst_path}")
        fastcp(src_path, dst_path, num_threads=4, buffer_size=2 * 1024 * 1024)

        # Verify files are identical
        with open(src_path, "rb") as src, open(dst_path, "rb") as dst:
            while True:
                src_chunk = src.read(8192)
                dst_chunk = dst.read(8192)

                if not src_chunk and not dst_chunk:
                    print("Verification successful: Files are identical.")
                    break

                if src_chunk != dst_chunk:
                    raise RuntimeError("Verification failed: Files are not identical.")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        # Clean up temporary files
        os.unlink(src_path)
        os.unlink(dst_path)
