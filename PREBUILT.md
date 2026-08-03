# Prebuilt Package Naming

Prebuilt engine packages are distributed as `.tar.gz` archives. The package name
must describe the engine, upstream revision, operating system, Ubuntu baseline,
and accelerator backend so tooling can select a compatible package
automatically.

## Package Name Format

```text
<engine>-<revision>-<os>-ubuntu<version>-<backend>.tar.gz
```

Example:

```text
llama.cpp-b9873-linux-ubuntu24.04-cuda.tar.gz
```

## Fields

`engine`
: Upstream engine project name. Current values are:
  `llama.cpp`, `parakeet.cpp`, and `kokoro.cpp`.

`revision`
: The upstream tag, branch, or commit identifier used to build the package.
  Examples: `b9873`, `db755a7`, `v0.1.0`.

`os`
: Target operating system. Current packages use `linux`.

`ubuntu<version>`
: Ubuntu release used as the compatibility baseline. Current values are:
  `ubuntu22.04` and `ubuntu24.04`.

`backend`
: Hardware backend. Current values are:
  `cpu` for CPU-only builds and `cuda` for NVIDIA CUDA builds.

## Current Package Examples

```text
kokoro.cpp-v0.1.0-linux-ubuntu22.04-cpu.tar.gz
kokoro.cpp-v0.1.0-linux-ubuntu24.04-cpu.tar.gz
llama.cpp-b9873-linux-ubuntu22.04-cpu.tar.gz
llama.cpp-b9873-linux-ubuntu22.04-cuda.tar.gz
llama.cpp-b9873-linux-ubuntu24.04-cuda.tar.gz
parakeet.cpp-db755a7-linux-ubuntu22.04-cuda.tar.gz
parakeet.cpp-db755a7-linux-ubuntu24.04-cuda.tar.gz
```

## Compatibility Rules

1. Package names must be lowercase, except where an upstream revision requires
   otherwise.
2. Use hyphens between fields. Do not add extra free-form text to the filename.
3. Use the exact engine project name, including `.cpp`.
4. Use `linux-ubuntu22.04` or `linux-ubuntu24.04` for Linux builds.
5. Use `cuda` only when the package requires NVIDIA CUDA at runtime.
6. Provide a `cpu` package when possible so systems without CUDA have a fallback.
7. Keep a `SHA256SUMS` file next to the archives and update it whenever a
   package changes.

## Interactive Installer

Use `prebuilt-cli.sh` to detect the current system, show compatible packages in
a numbered table, and install the selected archive:

```bash
./prebuilt-cli.sh
```

By default, the script reads the public Google Drive folder:

```text
https://drive.google.com/drive/folders/1B7AmE36r869kpMZbOatqMW-Dhedq2Sil?usp=sharing
```

For Google Drive access, the script uses Python 3 standard-library networking.
No extra Python packages are required.

```bash
./prebuilt-cli.sh
```

When a Drive URL is used, the script lists the public folder, shows the
compatible package table, and downloads only the selected archive into
`${XDG_CACHE_HOME:-$HOME/.cache}/zallama/prebuilt/downloads`. To use an already
downloaded package directory instead:

```bash
./prebuilt-cli.sh --source /path/to/packages
```

To restrict the table to one engine:

```bash
./prebuilt-cli.sh --engine llama.cpp
```

To install into another binary directory:

```bash
sudo ./prebuilt-cli.sh --bin-dir /opt/zallama/bin
```

To only print the compatible table without installing:

```bash
./prebuilt-cli.sh --list-only
```

The script detects:

- Operating system with `uname`
- Ubuntu version from `/etc/os-release`
- CUDA availability from `nvidia-smi`, `nvcc`, or `/usr/local/cuda`

Selection priority:

1. Exact Ubuntu version and detected backend
2. Exact Ubuntu version and `cpu`
3. Older compatible Ubuntu baseline and detected backend
4. Older compatible Ubuntu baseline and `cpu`

For example, on Ubuntu 24.04 with CUDA, `llama.cpp` prefers:

```text
llama.cpp-*-linux-ubuntu24.04-cuda.tar.gz
```

If no CUDA package exists for that engine and version, it falls back to a CPU
package when available.
