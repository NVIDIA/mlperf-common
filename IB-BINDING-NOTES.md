# Notes: per-rank NIC binding, and whether MPI could replace NCCL in datastage

Working notes for a follow-up session that has a real GPU node. Everything in
"What the code does" is read off `client/bindpcie` and is solid; everything in
"Predictions" is inference from reading and **has not been run on hardware**.
The point of the on-node session is to settle the predictions.

Delete this file once the questions are answered.

## Why this matters

`mlperf_common/fileio/datastage.py` uses NCCL, via torch, purely as a network
transport. Trace the bytes: Lustre → host → **GPU → fabric → GPU** → host →
NVMe. The data is host-resident at both ends and nothing computes on it, so the
GPU round trip costs two PCIe crossings a host-side transfer would not pay.

The justification in datastage's module docstring is that NCCL drives every NIC
"without any hand-tuned per-cluster transport configuration". That is an
argument about **NIC aggregation**, not about GPUs. If an MPI program can get
one NIC per rank — eight ranks, eight NICs, host memory — it should aggregate
the same fabric over a shorter path, and datastage could drop CUDA entirely.

What that would buy, beyond the shorter path:

* No torch, so no 20 GB container for a program that copies files. (Matt: the
  container copy cost is real but livable, so this is a secondary benefit, not
  the driver.)
* The whole class of bug fixed in `9035aa4` and `4ab9a13` — drainer thread
  device affinity, stream ordering, CUDA event semantics — stops existing,
  because there are no CUDA events.

What it risks: NCCL's out-of-the-box multi-NIC behaviour is genuinely good and
uniform across systems; MPI quality varies by site. That is a measurement, not
an argument, which is what this file is for.

## What `bindpcie --ib=single` actually does

All line numbers are `client/bindpcie`.

* **Device list** (`:107-111`): `ibv_devinfo --list | tail -n+2 | cut -f2`,
  in whatever order `ibv_devinfo` prints. `num_ibdevs` is that count.
* **GPU count** (`:79`): `nvidia-smi -i 0 --query-gpu=count`, so it reflects
  what is *visible* to the container.
* **Guard** (`:184-187`): if `num_ibdevs > num_gpus` or
  `num_gpus % num_ibdevs != 0`, print an error naming
  `MELLANOX_VISIBLE_DEVICES` and **`exit 1`**. A hard failure, not a warning.
* **Mapping** (`:189`):
  ```bash
  ibdev="${ibdevs[$(( local_rank * num_ibdevs / num_gpus ))]}"
  ```
* **Exports** (`:190-191`):
  ```bash
  export OMPI_MCA_btl_openib_if_include="${OMPI_MCA_btl_openib_if_include-$ibdev}"
  export UCX_NET_DEVICES="${UCX_NET_DEVICES-$ibdev:1}"
  ```

Note `MELLANOX_VISIBLE_DEVICES` is only ever *mentioned*, in the error message.
The script never reads it. It is an enroot/pyxis hook that filters which IB
devices the container sees, so it acts on `ibv_devinfo`'s output upstream of
this code.

## Predictions to verify on hardware

**1. The "near its GPU" claim is not implemented.** `--help` says `--ib=single`
binds "each rank to a single IB device near its GPU", but the mapping is pure
index arithmetic over `ibv_devinfo` order. There is no topology query anywhere
in the IB path — contrast the CPU path, which does interrogate `nvidia-smi` and
`lscpu`. Locality holds only if `ibv_devinfo` enumerates in GPU order.

This is the finding that would make the flag *harmful* rather than merely
useless: a mis-ordered list pins each rank to a NIC that may be across the root
complex, which is worse than letting UCX choose by locality.

With `num_ibdevs == num_gpus` the mapping reduces to `ibdevs[local_rank]`, so
the whole question collapses to: **does `ibv_devinfo --list` order match GPU
order on this platform?** That is directly checkable against
`nvidia-smi topo -m`.

**2. The guard probably passes under enroot, and probably would not bare.** A
stock H100 node reports compute *and* storage NICs to `ibv_devinfo` — typically
8 + 2 against 8 GPUs, so `10 > 8` and the script exits 1. Matt notes enroot does
set `MELLANOX_VISIBLE_DEVICES`, which should filter to the compute NICs and make
`num_ibdevs == 8`. Worth recording what `num_ibdevs` actually is in both
contexts rather than assuming.

**3. One of the two exported variables is dead.**
`OMPI_MCA_btl_openib_if_include` targets the openib BTL, deprecated in OpenMPI
4.0 and removed in 5.0. On anything modern it is ignored, and `UCX_NET_DEVICES`
is doing all the work.

**4. The UCX port is hardcoded** to `:1`. Correct for single-port cards, wrong
for a dual-port card whose second port carries the traffic.

**5. Diagnostics go to stdout.** `:183`, `:185`, `:186`, `:197` all use
`echo "..." 2>&1`, which is a no-op for `echo` — the intent was `>&2`. Compare
`:80`, which gets it right. So these errors and warnings land on **stdout** and
will interleave with the wrapped program's own output. Minor, but it is one
reason a broken `--ib=single` could go unnoticed for years.

## What to check on the node

```bash
# 1. What does the container actually see?
echo "MELLANOX_VISIBLE_DEVICES=${MELLANOX_VISIBLE_DEVICES:-unset}"
ibv_devinfo --list
nvidia-smi -i 0 --query-gpu=count --format=csv,noheader,nounits

# 2. Does ibv_devinfo order match GPU order?  This is the crux of prediction 1.
nvidia-smi topo -m          # look for PIX/PXB between GPU i and each NIC
#    then compare against ibdevs[i] for i in 0..num_gpus-1

# 3. Does the binding take at all?
srun ... bindpcie --ib=single -- bash -c 'echo "$SLURM_LOCALID $UCX_NET_DEVICES"'
```

## Observing NIC usage — do not infer it from bandwidth

Ground truth, per device, independent of what any library claims:

```bash
cat /sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data
```

Sample before and after a transfer and diff. That answers "did all eight NICs
move bytes" directly. (Units are 4-byte lanes, which does not matter for a
did-it-move check.)

For the other half — "did *this rank* select the NIC we told it to" —
`UCX_LOG_LEVEL=info`, or `UCX_PROTO_INFO=y` on newer UCX, prints each rank's
selected transports and devices. For OpenMPI, `--mca pml_ucx_verbose 10`.

**Caveat for the comparison:** UCX defaults to `UCX_MAX_RNDV_RAILS=2`, so an
*unbound* rank may already use two NICs for large transfers. Pinning each rank
to exactly one device can therefore lower per-rank bandwidth while raising
aggregate spread. Measure **aggregate across all eight ranks**, bound versus
unbound — a single rank's number will mislead.

## The question this feeds

If eight ranks each on their own NIC, in host memory, get within ~10% of what
datastage's current all-gather achieves (its per-file `GB/s` line, or the final
`DONE ... GB/s`), then the GPU is pure cost in this program and an MPI transport
is strictly better: shorter path, smaller container, and a large category of
correctness surface deleted.

If they do not, the NCCL path is earning its keep and the right move is instead
to drop *torch* while keeping NCCL — `tests/stubs.py` is already an interface
specification for exactly the surface datastage uses (`torch.empty`, `device`,
`cuda.{Event,Stream,stream,current_stream,current_device,set_device,
get_device_properties}`, `dist.{barrier,get_rank,new_group,
broadcast_object_list,all_gather_into_tensor}`), and a ctypes NCCL backend is a
second implementation of it. The rendezvous falls out too: datastage always has
a shared filesystem, so rank 0 can drop the 128-byte `ncclUniqueId` in a file
instead of standing up a `TCPStore`.
