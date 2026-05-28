# hf-mount cache examples

> ⚠️ **READ THIS BEFORE TOUCHING ANY HF-MOUNT CODE OR MOUNT POINT.**
> Mistakes here brick the system. Recovery is **reboot only**. The
> user has had to reboot multiple times because agents ignored these
> rules. Do not assume "just this once" is safe.

## hf-mount lifecycle — non-negotiable

There are exactly **two** ways to operate an hf-mount mount:

✅ **DO** — use the `hf-mount` wrapper, which manages the daemon and
performs a coordinated unmount on stop:

```bash
hf-mount start -- <backend flags> bucket <BUCKET> <MOUNTPOINT>
hf-mount stop  <MOUNTPOINT>
hf-mount status                # list running daemons
```

❌ **DO NOT** — any of the following will leave the kernel with a
stale NFS mount whose daemon is gone (phantom mount). Any later
`stat` / `ls` / `du` / `rm` / `find` / `df` on the path will block
forever on NFS retransmits and propagate the wedge to every process
that touches it. **Reboot is the only recovery.**

1. `umount <MOUNTPOINT>` — even `sudo`, even `-l`, even `-f`.
2. `fusermount -u <MOUNTPOINT>` — same outcome.
3. Launching `hf-mount-nfs` (or `hf-mount-fuse`) **directly** as a
   background process. The backend binaries do **not** unmount on
   SIGTERM. Killing them strands the mount.
4. `kill` / `pkill` of an hf-mount PID. Same as (3).
5. Trusting old docs / comments that say "send SIGTERM to hf-mount
   and it handles its own unmount." That advice predated the
   wrapper and is wrong for the backend binaries.

### Phantom-mount diagnosis

If you suspect a stale mount (commands hanging on `/tmp/hf-mount-*`,
or you see a mount in `/proc/mounts` that wasn't there a moment ago):

```bash
hf-mount status                       # any daemons?
grep hf-mount /proc/mounts            # mount entries?
ss -tln | grep <port-from-mount>      # listener on the NFS port?
```

If `/proc/mounts` shows the mount but `hf-mount status` reports no
daemons and the NFS port has no listener, **stop immediately**.
Don't try `umount`, don't `ls` the mount, don't `rm` anything under
it. Tell the user. Recovery is reboot.

## What the example does

Single integration test: [torch.compile/](torch.compile/) — shares
the Inductor on-disk compile cache through an HF bucket so cold
starts on the consumer side skip the compile.

### Phases

| Phase     | hf-mount flags                              | Purpose                                    |
|-----------|---------------------------------------------|--------------------------------------------|
| `warmup`  | `--advanced-writes` (RW, async batched flush) | Populate the bucket                        |
| `consume` | `--overlay` (implies `--advanced-writes`)     | Lazy-fetch from bucket; writes stay local  |

`--advanced-writes` is **required** for the producer (warmup). Without
it, hf-mount runs append-only with synchronous close: every Inductor
cache-file close blocks on upload, and warmup hangs for hours.

### Compile path

`compile_run.py` uses transformers' static-cache compile path:

```python
model.generation_config.cache_implementation = "static"
model.generate(input_ids, cache_implementation="static", ...)
```

Setting `cache_implementation="static"` is what triggers transformers'
automatic compile. We rely on transformers' default compile mode —
swapping to `max-autotune` variants explodes the Inductor artifact
count and tanks bucket-sync time. See "Caveats" in
[torch.compile/README.md](torch.compile/README.md).

## Running

```bash
cd torch.compile && ./setup.sh                  # one-time
source ../.venv/bin/activate
./run.sh clear-bucket                  # optional clean slate
./run.sh run-all                       # warmup + consume
```

Individual commands: `warmup`, `consume`, `teardown`, `clear-bucket`.

## Repo layout

```
hf-mount-cache-examples/
├── README.md
├── AGENTS.md                # this file
├── torch.compile/
│   ├── setup.sh             # installs hf-mount + venv + torch/transformers
│   ├── run.sh               # CLI: warmup / consume / teardown
│   ├── compile_run.py       # load model, torch.compile, generate, time it
│   └── README.md
└── .venv/                   # shared Python venv (../.venv from torch.compile)
```
