///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <thread>
#include <atomic>               // for atomic_int
#include <vector>
#include <algorithm>            // for min
#include <memory>               // for std::unique_ptr

#include <sys/types.h>          // for open/creat/stat
#include <sys/stat.h>           // for open/creat/stat
#include <sys/mman.h>           // for mmap
#include <fcntl.h>
#include <unistd.h>             // for stat, pread, pwrite, lseek, close
#include <stdio.h>              // for perror
#include <sys/statvfs.h>        // for statvfs (to get fs blocksize for direct
                                // reads)

#include "cmdline.h"

constexpr ssize_t KiB = 1024;
constexpr ssize_t MiB = KiB * KiB;
constexpr ssize_t GiB = KiB * MiB;
constexpr ssize_t TiB = KiB * GiB;
constexpr ssize_t PiB = KiB * TiB;

constexpr ssize_t minBufSize = 2 * MiB;

struct CmdLineArgs {
    cmdline::Param<bool>    help       {"-?", "--help",
                                        "this message"};
    cmdline::Param<ssize_t> bufSize    {"-b", "--buffer-size",
                                        "buffer size",         16*MiB};
    cmdline::Param<int>     numThreads {"-n", "--num-threads",
                                        "number of threads",
                                        static_cast<int>(std::thread::hardware_concurrency())};
};

std::string usageString=
    "Usage: fastcp [ARGS] INFILE OUTFILE"
    ;

///////////////////////////////////////////////////////////////////////////////
// aligned memory buffer (unqiue_ptr wrapper for posix_memalign()
///////////////////////////////////////////////////////////////////////////////

struct FreeDeleter
{
    void operator()(void* ptr) const {
        free(ptr);
    }
};

typedef std::unique_ptr<void, FreeDeleter> UniqueBuffer;

UniqueBuffer
createAlignedUniqueBuffer(size_t alignment, size_t size)
{
    void* buffer = nullptr;
    int memalign_error = posix_memalign(&buffer, alignment, size);
    if (memalign_error != 0) {
        if (memalign_error == EINVAL) {
            std::cerr
                <<
                "createAlignedUniqueBuffer requires alignment power of 2";
        } else if (memalign_error == ENOMEM) {
            std::cerr
                << "createAlignedUniqueBuffer out of memory";
        } else {
            std::cerr
                << "createAlignedUniqueBuffer unknown error "
                << memalign_error;
        }
        std::cerr << std::endl;
        exit(1);
    }

    return UniqueBuffer(buffer);
}


///////////////////////////////////////////////////////////////////////////////
// The simplest possible workpile algorithm we could think of for load
// balancing. Just launch numThreads workers on numChunks items of work and
// each worker uses pile->fetch_add(1) to fetch the next unit of work
///////////////////////////////////////////////////////////////////////////////

typedef std::atomic<int> pile_type;

///////////////////////////////////////////////////////////////////////////////
// abstractions over fstat
///////////////////////////////////////////////////////////////////////////////

off_t
getFileSize(int fD)
{
    struct stat statbuf;

    if (fstat(fD, &statbuf) != 0) {
        perror("error calling stat on input file");
        return -1;
    }

    return statbuf.st_size;
}

off_t
getFileSystemBlockSize(int fD)
{
    struct statvfs statvfsbuf;

    if (fstatvfs(fD, &statvfsbuf) != 0) {
        perror("error calling statvfs to get block size of input file");
        return -1;
    }

    return statvfsbuf.f_bsize;
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
constexpr T
ceilDiv(T dividend, T divisor)
{
    static_assert(std::is_integral<T>::value, "Integer type required.");
    return (dividend+(divisor-1))/divisor;
}

template <typename T>
constexpr T
roundUp(T value, T multiple)
{
    static_assert(std::is_integral<T>::value, "Integer type required.");
    return ceilDiv(value, multiple)*multiple;
}

template <typename T>
constexpr bool
isPowerOfTwo(T x)
{
    static_assert(std::is_integral<T>::value, "Integer type required.");
    return (x > 0) && ((x & (x-1)) == 0);
}

// much like pread()
//
// read up to chunkSize bytes from file descriptor fd at offset offset (from
// the start of the file) into the buffer starting at buf.  The file offset is
// not changed.
//
// on success, returns the number of bytes read
//
// The extra arguments fileSize and inputFSBlockSize are required in order to
// correctly handle O_DIRECT reads for the last block of a file.  The extra
// argument threadId is just used to make the error reporting clearer.
off_t
readFileChunk(int fd,
              void* buf,
              off_t chunkSize,
              off_t offset,
              off_t fileSize,
              off_t inputFSBlockSize,
              int threadId)
{
    assert(inputFSBlockSize > 0);
    assert(chunkSize > 0);
    assert(offset + chunkSize <= fileSize);

    // O_DIRECT read size for last block needs to be very very specific: it
    // needs to be an exact multiple of the filesystem block size on which
    // the file resides, but it can't be more than a single block larger!
    off_t myReadSize = roundUp(chunkSize, inputFSBlockSize);
    // alignment requirements for ODIRECT reads:
    assert(myReadSize % inputFSBlockSize == 0);
    assert(reinterpret_cast<intptr_t>(buf) % inputFSBlockSize == 0);
    assert(offset % inputFSBlockSize == 0);
    if (myReadSize != chunkSize) {
        std::cerr << "thread " << threadId
                  << " is about to read the last (partial) block of the input file"
                  << std::endl
                  << "filesize is "
                  << fileSize
                  << std::endl
                  << "offset is "
                  << offset
                  << std::endl
                  << "request size is "
                  << chunkSize
                  << std::endl
                  << "read size is "
                  << myReadSize
                  << std::endl;
    }
    do {
        off_t readResult = pread(fd,
                                 buf,
                                 myReadSize,      // size of read
                                 offset);         // file offset

        // with O_DIRECT even though we need to ask for the rounded up size,
        // the number of bytes we get should be the number left in the file,
        // not extra garbage at the end of the rounded up block.
        if (readResult == chunkSize) {
            // expected case - done with loop!
            return readResult;
        }
        assert (readResult < chunkSize);

        // noisy failure cases
        if (readResult == 0) {
            // unexpected EOF
            return readResult;
        }
        if ((readResult < 0) && (errno != EINTR)) {
            // error other than EINTR
            return readResult;
        }

        // retry cases (incomplete read, or EINTR)
        std::cerr << "thread " << threadId
                  << " got partial or interrupted read of size " << readResult
                  << " when trying to read offset " << offset
                  << ": retrying"
                  << std::endl;
    } while (1);
    return -1;
}

// much like pwrite()
//
// write up to chunkSize bytes from file descriptor fd at offset offset (from
// the start of the file) into the buffer starting at buf.  The file offset is
// not changed.
//
// on success, returns the number of bytes written
//
// This is intended for use with O_DIRECT, so write sizes are rounded up to the
// next multiple of outputFSBlockSize.  (Truncate the file to the correct
// length after finishing the writes).
//
// The extra argument threadId is just used to make the error reporting
// clearer.
off_t
writeFileChunk(int fd,
               void* buf,
               off_t chunkSize,
               off_t offset,
               off_t outputFSBlockSize,
               int threadId)
{
    assert(outputFSBlockSize > 0);
    assert(chunkSize > 0);

    // O_DIRECT write size for last block needs to be an exact multiple of the
    // filesystem block size on which the file resides.  (We take care of any
    // extra garbage written in the last block, by later using truncate to trim
    // the file back to its correct size.)
    off_t myWriteSize = roundUp(chunkSize, outputFSBlockSize);
    // alignment requirements for ODIRECT writes:
    assert(myWriteSize % outputFSBlockSize == 0);
    assert(reinterpret_cast<intptr_t>(buf) % outputFSBlockSize == 0);
    assert(offset % outputFSBlockSize == 0);
    do {
        off_t writeResult = pwrite(fd,
                                   buf,
                                   myWriteSize,      // size of read
                                   offset);         // file offset

        if (writeResult == myWriteSize) {
            // expected case - done with loop!
            return writeResult;
        }
        assert (writeResult < myWriteSize);

        // noisy failure case:
        if ((writeResult < 0) && (errno != EINTR)) {
            // error other than EINTR
            return writeResult;
        }

        // retry cases (incomplete write, or EINTR)
        std::cerr << "thread " << threadId
                  << " got partial or interrupted write of size " << writeResult
                  << " when trying to write offset " << offset
                  << ": retrying"
                  << std::endl;
    } while (1);
    return -1;
}

void
worker(pile_type* pile,
       int threadId,
       int inFD,
       int outFD,
       ssize_t bufferSize,
       off_t maxChunks,
       off_t fileSize,
       off_t inputFSBlockSize,
       off_t outputFSBlockSize)
{
    UniqueBuffer myBuf = createAlignedUniqueBuffer(std::min(bufferSize, minBufSize), bufferSize);
    for (int i = pile->fetch_add(1); i < maxChunks; i = pile->fetch_add(1)) {
        off_t myOffset = i*bufferSize;
        off_t mySize = std::min(bufferSize, fileSize-myOffset);

        auto readResult = readFileChunk(inFD,
                                        myBuf.get(),
                                        mySize,
                                        myOffset,
                                        fileSize,
                                        inputFSBlockSize,
                                        threadId);
        if (readResult != mySize) {
            if (readResult == 0) {
                perror("unexpected EOF from readFileChunk()");
            } else {
                perror("failure calling readFileChunk()");
            }
            continue;
        }
        else {
            // succesful read: now write the buffer to the output
            off_t writeResult = writeFileChunk(outFD,
                                               myBuf.get(),
                                               mySize,
                                               myOffset,
                                               outputFSBlockSize,
                                               threadId);
            if (writeResult < mySize) {
                if (writeResult >= 0) {
                    perror("unexpected partial write from writeFileChunk()?");
                } else {
                    perror("failure calling writeFileChunk()");
                }
            }
        }
    }
}

int
main(int argc,
     char* argv[])
{
    CmdLineArgs args;
    int argsProcessed = cmdline::processArgs(argc, argv);

    if (args.help) {
        std::cout << usageString << std::endl;
        std::cout << cmdline::helpMsg() << std::endl;
        exit(0);
    }

    if ((argc - argsProcessed) != 2) {
        std::cerr << "Error: must provide exactly one input file and one output file" << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    if ((args.bufSize % minBufSize) != 0) {
        std::cerr << "Error: --buf-size must be a multiple of 2MiB ("
                  << minBufSize << ")"
                  <<std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    const std::string inFileName{argv[argc-2]};
    const std::string outFileName{argv[argc-1]};

    int inFD = open(inFileName.c_str(), O_RDONLY|O_DIRECT);
    if (inFD < 0) {
        perror("opening input direct failed, retrying indirect");
        inFD = open(inFileName.c_str(), O_RDONLY);
        if (inFD < 0) {
            perror("opening input file failed");
            exit(1);
        }
    }
    off_t fileSize = getFileSize(inFD);
    off_t inFsBlockSize = getFileSystemBlockSize(inFD);
    if (fileSize < 0) {
        std::cerr
            << "Error: could not find valid file size for input file "
            << inFileName
            << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    off_t maxChunks = ceilDiv(fileSize, static_cast<off_t>(args.bufSize));
    std::cout << "infile name is " << inFileName << std::endl;
    std::cout << "outfile name is " << outFileName << std::endl;
    std::cout << "file size is " << fileSize << std::endl;
    std::cout << "the read size is " << args.bufSize << std::endl;
    std::cout << "the input fs block size is " << inFsBlockSize << std::endl;
    std::cout << "max chunks is " << maxChunks << std::endl;
    std::cout << "the number of threads will be " << args.numThreads << std::endl;

    int outFD = open(outFileName.c_str(),
                     O_CREAT|O_RDWR|O_DIRECT,
                     S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);
    if (outFD < 0) {
        perror("opening outfile failed");
        return (1);
    }
    off_t outFsBlockSize = getFileSystemBlockSize(outFD);
    std::cout << "the output fs block size is " << outFsBlockSize << std::endl;

    std::cerr << "starting threads" << std::endl;
    pile_type the_pile{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < args.numThreads; i++) {
        workers.push_back(std::thread(worker,
                                      &the_pile,
                                      i,
                                      inFD,
                                      outFD,
                                      args.bufSize,
                                      maxChunks,
                                      fileSize,
                                      inFsBlockSize,
                                      outFsBlockSize));
    }
    // wait for everyone to finish
    for (std::thread &t: workers) {
        if (t.joinable()) {
            t.join();
        }
        else {
            std::cerr << "internal error: unjoinable thread???" << std::endl;
        }
    }

    if (close(inFD) != 0) {
        perror("close of input file failed");
        // don't exit, keep trying to finish cleanup
    }
    if (ftruncate(outFD, fileSize) != 0) {
        perror("ftruncate of output file failed");
        return (1);
    }

    if (close(outFD) != 0) {
        perror("close of output file failed");
        exit(1);
    }

    return 0;
}
