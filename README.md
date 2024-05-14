# MLPerf Common - a collection of common MLPerf tools


## MLPerf Logging

MLPerf common can be installed via `pip install` by adding the following line to the `requirements.txt` file:
```
git+https://github.com/NVIDIA/mlperf-common.git
```

### Integration using torch.distributed (pytorch)

In `mlperf_logger.py` module define:
```
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)
```
Then use `mllogger` by importing `from mlperf_logger import mllogger` in other modules.

### Integration using MPI (horovod/hugectr/mxnet/tensorflow)

In `mlperf_logger.py` global module define:
```
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.mxnet import MPICommunicationHandler

mllogger = MLLoggerWrapper(MPICommunicationHandler(), value=None)
```
Then use `mllogger` by importing `from mlperf_logger import mllogger` in other modules.

Optionally, you can pass an MPI communicator during the initialization of `MPICommunicationHandler()`.
```
comm = MPI.COMM_WORLD
mllogger = MLLoggerWrapper(MPICommunicationHandler(comm), value=None)
```
by default, `MPICommunicationHandler()` creates a global communicator.

### Logging additional metrics
MLPerf logger can be used to track additional non-required metric, for example `throughput`. The recommended way is to add a line such as:
```
mllogger.event(key='tracked_stats', metadata={'step': epoch}, value={"throughput": throughput, "metric_a": metric_a, 'metric_b': metric_b})
```
where `throughput` is recommended to be `samples per second`, logged every epoch or as often as it is reasonable for a given benchmark. Additional metrics, `metric_a` and `metric_b`, can represent any numerical value that requires logging. The key `tracked_stats` and an increasing value for `step` are required.

## Scaleout Bridge

#### init_bridge
Instead of previous `sbridge = init_bridge(rank)`, initialize sbridge as follows:
```
from mlperf_common.frameworks.pyt import PyTNVTXHandler, PyTCommunicationHandler

sbridge = init_bridge(PyTNVTXHandler(), PyTCommunicationHandler(), mllogger)
```
or, for `horovod/tf/mxnet`:
```
from mlperf_common.frameworks.mxnet import MXNetNVTXHandler, MPICommunicationHandler

sbridge = init_bridge(MXNetNVTXHandler(), MPICommunicationHandler(), mllogger)
```
and start your profiling as usual
```
sbridge.start_prof()
sbridge.stop_prof()
```

#### EmptyObject
Current `ScaleoutBridgeBase` class replaces previous `EmptyObject` class,
so just replace `EmptyObject()` with `ScaleoutBridgeBase()`.

## Mount check
### Get mount info
```
from mlperf_common.mountcheck import getMountInfo
verifyMount(paths_to_verify)
```
`verifyMount(paths_to_verify)` prints a JSON with description of given directories.  
`paths_to_verify` should contain paths separated by commas.  

`number_of_paths_in_dir` for each path specified contains the number of subdirectories in it.  
The fields in directory_sizes contain 3 values separated by commas:  
relative path path to the directory, its size in KB, the number of directories and files inside it.



### Verify mounts
`verify-mounts.sh` checks if a given directory is consistent with a description generated with get-mount-info.sh and takes one argument:
`paths_to_verify` that contains paths separated by commas.  

Example of use:
```
verify-mounts.sh "/data,/checkpoints"
```
The directory where `verify-mounts.sh` is located should contain `cont-mount-info.sh` file generated earlier by `get-mount-info.sh`.
