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
`mountcheck.py` is a script that allows to create a JSON file with the description of the given paths and later verify the correctness of these paths based on the information from the JSON file.  
The interface is as follows:
```
mountcheck.py [--check] [PATH]
```

### Get mount info
Usage:
```
./mountcheck.py --check /path/to/verify/1 path/to/verify/2 ... path/to/verify/n
```
In this mode `mountcheck.py` generates a description of the given directories.  
`/path/to/verify1 path/to/verify/2 ... path/to/verify/n` is a list of directories whose description is to be created.  
The description will be written out to the standard output in a JSON format.


### Verify mounts
Usage:
```
./mountcheck.py /path/to/file.json
```
In this mode `mountcheck.py` checks if a given directories are consistent with given JSON file.  
`/path/to/file.json` abc is a JSON file with a previously generated description of the paths being verified. 