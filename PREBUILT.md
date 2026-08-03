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

## Automatic Selection

Use `find-compatible-prebuilt.sh` to choose the best archive from a local folder
containing downloaded packages:

```bash
./find-compatible-prebuilt.sh /path/to/packages
```

To restrict selection to one engine:

```bash
./find-compatible-prebuilt.sh /path/to/packages llama.cpp
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
