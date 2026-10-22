# Native Atari container contract

The user requested a GPU container and then directed the implementation to use
C/C++ instead of Python. This native contract supersedes the unbuilt Python
proposal. Creation follows this document and `atari.ps1`.

The initial never-started container encountered a conflict on port 43118.
The user approved replacing that exact container on port 43260 while preserving
its image and bind-mounted data. The table records the corrected configuration;
see [VALIDATION.md](VALIDATION.md) for the original inspection and recovery record.

| Setting | Value |
| --- | --- |
| Container | `atari-dev` |
| Image | `atari:dev` |
| Base | `debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171` |
| Command | `sh /workspace/run.sh` |
| Application | `/workspace/build/native/atari serve --root /workspace --rom /opt/ale/roms/pong.bin --host 0.0.0.0 --port 8080` |
| Working directory | `/workspace` |
| Bind | `C:\Work\Natural-Dexterity\projects\manual-dexterity\atari` to `/workspace`, read-write |
| Named volumes | None |
| Port | `127.0.0.1:43260` to `8080/tcp` |
| Restart | `no` (manual start only) |
| GPU | `--gpus all`, NVIDIA compute and utility capabilities |
| Other devices | None |
| Init/shared memory/stop timeout | Enabled / 1 GiB / 30 seconds |
| Labels | `io.waajacu.managed=true`, `io.waajacu.project=atari`, `io.waajacu.config=native-v1` |

`setup.sh` installs native dependencies only, accepts no operation arguments,
and uses `--no-install-recommends` for APT. Native archives have pinned hashes
in `dependencies.lock`. Python is neither installed nor used to build, run,
learn, evaluate, or serve the application.

`run.sh` builds the project and starts the native viewer. `atari.ps1` inspects
and reuses an exact matching container, starting a stopped one by immutable ID.
A conflict is preserved. No command deletes containers, images, volumes,
caches, or experiments. Runs and checkpoints stay on the host bind mount.

Automatic startup was disabled at the user's request on 2026-09-20 by updating
the existing container's restart policy in place. Start it when needed with
`atari.ps1 start`; starting Docker does not start this container.
