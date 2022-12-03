///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <sys/types.h>          // for open/creat/stat
#include <sys/stat.h>           // for open/creat/stat
#include <sys/mman.h>           // for mmap
#include <fcntl.h>
#include <unistd.h>             // for stat, pread, pwrite, lseek, close
#include <stdio.h>              // for perror
#include <string.h>             // for memset

#include "cmdline.h"

const ssize_t KiB = 1024;
const ssize_t MiB = KiB * KiB;
const ssize_t GiB = KiB * MiB;
const ssize_t TiB = KiB * GiB;
const ssize_t PiB = KiB * TiB;

struct CmdLineArgs {
    cmdline::Param<bool>    help       {"-?", "--help",
                                        "this message"};
    cmdline::Param<ssize_t> bufSize    {"-b", "--buffer-size",
                                        "buffer size",         16*MiB};
    cmdline::Param<int>     numThreads {"-n", "--num-threads",
                                        "number of threads",
                                        static_cast<int>(std::thread::hardware_concurrency())};
    cmdline::Param<ssize_t> outSize    {"-s", "--size",
                                        "output file size", -1};
};

std::string usageString=
    "Usage: create-empty-file --size N [ARGS] OUTFILE\n"
    "typically you would calculate size with something like `$(stat --format=%s FILENAME)'"
    ;
typedef std::atomic<int> pile_type;

off_t
ceilDiv(off_t dividend, off_t divisor)
{
    return (dividend+(divisor-1))/divisor;
}

void
worker(pile_type* pile,
       int threadId,
       void* outBuf,
       ssize_t bufferSize,
       off_t fileSize)
{
    off_t maxChunks = ceilDiv(fileSize, bufferSize);
    for (int i = pile->fetch_add(1); i < maxChunks; i = pile->fetch_add(1)) {
        off_t myOffset = i*bufferSize;
        if (myOffset >= fileSize) break;
        assert((fileSize-myOffset) > 0);
        off_t mySize = std::min(bufferSize, fileSize-myOffset);
        memset(static_cast<char*>(outBuf)+myOffset, 'f', mySize);
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

    if ((argc - argsProcessed) != 1) {
        std::cerr << "Error: must provide exactly one output file" << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    const std::string outFileName{argv[argc-1]};
        
    off_t outSize = args.outSize;
    if (outSize <= 0) {
        std::cerr
            << "Error: must provide --size argument"
            << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    std::cout << "outfile name is " << outFileName << std::endl;
    std::cout << "output size is " << outSize << std::endl;
    std::cout << "the buffer size is " << args.bufSize << std::endl;
    std::cout << "the number of threads will be " << args.numThreads << std::endl;

    int outFD = open(outFileName.c_str(),
                     O_CREAT|O_RDWR,
                     S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);
    if (outFD < 0) {
        perror("opening outfile failed");
        return (1);
    }
    if (ftruncate(outFD, outSize) != 0) {
        perror("ftruncate of output file failed");
        return (1);
    }

    // we specifically do NOT use MAP_POPULATE here because it seems to lead to
    // buffers that are slower to random access later
    void* outBuf = mmap(NULL, outSize, PROT_READ|PROT_WRITE, MAP_SHARED,
                        outFD, 0);
    if (outBuf == MAP_FAILED) {
        perror("mmap to allocate memory for cache failed");
        return (1);
    }

    std::cerr << "starting threads" << std::endl;
    pile_type the_pile{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < args.numThreads; i++) {
        workers.push_back(std::thread(worker,
                                      &the_pile,
                                      i,
                                      outBuf,
                                      args.bufSize,
                                      outSize));
    }
    // wait for everyone to finish
    for (std::thread &t: workers) {
        if (t.joinable()) {
            t.join();
        }
        else {
            std::cerr << "unjoinable thread???" << std::endl;
        }
    }

    if (close(outFD) != 0) {
        perror("close of output file failed");
        exit(1);
    }

    return 0;
}
