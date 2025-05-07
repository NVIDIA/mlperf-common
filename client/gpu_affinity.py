import os
import math                     # for ceil()
import pynvml                   # from package nvidia-ml-py3

################################################################################ 
# start python script bound to appropriate cores for gpu given by the
# LOCAL_RANK environment variable
#
# it is assumed we are running under slurm (with pyxis/enroot) and the
# following variables are set in the environment and will be passed to
# pytorch.distributed
#
# LOCAL_RANK: the gpu to use on the local node RANK: slurm_ntasks_per_node *
# slurm_nodeid + local_rank WORLD_SIZE: slurm_ntasks_per_node * slurm_nnodes
# MASTER_ADDR: hostname of the node with rank 0 MASTER_PORT: tcp/ip port on
# node MASTER_ADDR
#
# and additionally we have:
#
# SLURM_NNODES (number of nodes) SLURM_NODEID (unique node number: 0 <=
# SLURM_NODEID < SLURM_NNODES)
#
# SLURM_NTASKS_PER_NODE (should be same as number of gpus on the node)
# SLURM_CPUS_ON_NODE (should be same as os.cpu_count())
#
# Note that the actual number of visible gpus is controlled by the value of
# NVIDIA_VISIBLE_DEVICES when srun is invoked, while SLURM_NTASKS_PER_NODE
# comes from the --ntasks-per-node flag to srun.
#
# We must carefully "by hand" make sure --ntasks-per-node is equal to the
# number of gpus in the NVIDIA_VISIBLE_DEVICES set on the host.
#
# If we do this correctly, On each node (inside the container) we will get
# SLURM_NTASKS_PER_NODE processes, with local ranks in
# [0...SLURM_NTASK_PER_NODE), and the gpus will have the same numbering.
# Inside the container only the appropriate gpus will be visible, and they will
# be numbered 0...SLURM_NTASKS_PER_NODE, and NVIDIA_VISIBLE_DEVICES will be
# "all", not the "real" list of gpus.
################################################################################ 

pynvml.nvmlInit()

def systemGetDriverVersion():
    return pynvml.nvmlSystemGetDriverVersion()

def deviceGetCount():
    return pynvml.nvmlDeviceGetCount()

class device:
    # assume nvml returns list of 64 bit ints
    _nvml_affinity_elements = math.ceil(os.cpu_count()/64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def getName(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string
        # "pythonic" way to turn string into list of 0/1 ints
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()     # so core 0 is in 0th element of list
            
        # "pythonic" way to turn into list of indices of non-zero elements
        return [i for i,e in enumerate(affinity_list) if e!=0]

def set_affinity(gpu_id=None):
    if gpu_id is None:
        gpu_id = int(os.environ.get('LOCAL_RANK'))
    #print(gpu_id, nvml.systemGetDriverVersion(), nvml.deviceGetCount())
    dev = device(gpu_id)
    #print(gpu_id, dev.getName())

    # I can't find the equivalent of numactl --membind with
    # sched_setaffinity(), but it seems that the default is equivale to numactl
    # --localalloc, which is what we really want (since we are always binding
    # to cores on only one socket)
    os.sched_setaffinity(0, dev.getCpuAffinity())

    # list of ints representing the logical cores this process is now affinitied with
    return os.sched_getaffinity(0)
