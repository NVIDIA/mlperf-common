# MLPerf Common - a collection of common MLPerf tools


## MLPerf Logging

Clone the repo in the Dockerfile:
```
RUN git clone --depth=1 --branch=main https://github.com/NVIDIA/mlperf-common mlperf-common-repo \
  && cp mlperf-common-repo/client/bindpcie . \
  && cp -r mlperf-common-repo/mlperf_common . \
  && rm -rf mlperf-common-repo
 ```

### Integration using torch.distributed (pytorch)

In `mlperf_logger.py` module define:
```
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)
```
Then use `mllogger` by importing `from mlperf_logger import mllogger` in other modules.

### Integration using MPI (horovod/mxnet/tensorflow)

In `mlperf_logger.py` global module define:
```
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.mxnet import MPICommunicationHandler

mllogger = MLLoggerWrapper(MPICommunicationHandler(), value=None)
```
Then use `mllogger` by importing `from mlperf_logger import mllogger` in other modules.

Optionally, you can pass an MPI communicator during the initialization of `MPICommunicationHandler()`
```
comm = MPI.COMM_WORLD
mllogger = MLLoggerWrapper(MPICommunicationHandler(comm), value=None)
```
by default, `MPICommunicationHandler()` creates a global communicator

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