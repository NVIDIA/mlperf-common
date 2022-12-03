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
    cmdline::Param<ssize_t> count      {"-c", "--count",
                                        "number of buffers to copy in each work item",
                                        1};
    cmdline::Param<int>     numThreads {"-n", "--num-threads",
                                        "number of threads",
                                        static_cast<int>(std::thread::hardware_concurrency())};
    cmdline::Param<bool>    direct     {"-d", "--direct",
                                        "Use direct file i/o", false};
    cmdline::Param<ssize_t> outSize    {"-s", "--size",
                                        "output file size", -1};
    cmdline::Param<std::string> inFile {"-i", "--input-file",
                                        "make output file same size as given input file",
                                        ""};
};

std::string usageString="Usage: create-empty-file [ARGS] OUTFILE";
typedef std::atomic<int> pile_type;

off_t
getFileSize(const std::string& filePath)
{
    struct stat statbuf;

    if (stat(filePath.c_str(), &statbuf) != 0) {
        perror("error calling stat on input file");
        return -1;
    }

    return statbuf.st_size;
}

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
       ssize_t count,
       off_t fileSize)
{
    off_t chunkSize = bufferSize*count;
    off_t maxChunks = ceilDiv(fileSize, chunkSize);
    for (int i = pile->fetch_add(1); i < maxChunks; i = pile->fetch_add(1)) {
        for (int j = 0; j < count; ++j) {
            off_t myOffset = (i*chunkSize) + (j*bufferSize);
            if (myOffset >= fileSize) break;
            assert((fileSize-myOffset) > 0);
            off_t mySize = std::min(bufferSize, fileSize-myOffset);
            memset(static_cast<char*>(outBuf)+myOffset, 'f', mySize);
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

    if ((argc - argsProcessed) != 1) {
        std::cerr << "Error: must provide exactly one output file" << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    const std::string outFileName{argv[argc-1]};
        
    off_t outSize = args.outSize;
    const std::string inFileName{args.inFile};
    if (inFileName != "") {
        if (outSize <= 0) {
            outSize = getFileSize(inFileName);
        }
        else {
            // ambiguous to specify both: error
            outSize = -1;
        }
    }
    
    if (outSize <= 0) {
        std::cerr
            << "Error: must provide exactly one of --size or --input-file to choose output file size"
            << std::endl;
        std::cerr << usageString << std::endl;
        std::cerr << cmdline::helpMsg() << std::endl;
        exit(1);
    }

    if (args.direct) {
        std::cerr
            << "Sorry: --direct flag not currently supported!"
            << std::endl;
        exit(1);
    }

    std::cout << "outfile name is " << outFileName << std::endl;
    std::cout << "output size is " << outSize << std::endl;
    std::cout << "the buffer size is " << args.bufSize << std::endl;
    std::cout << "the number of threads will be " << args.numThreads << std::endl;
    std::cout << "writing in chunks of " << args.count << " buffers" << std::endl;

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
                                      args.count,
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
