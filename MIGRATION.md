# Migration Notes

## 0.2.0: tensor-runtime-neutral file-format core

0.2.0 intentionally removes the legacy `Crucible.Safetensors.Slice` API and the
package's direct Nx / external Safetensors dependencies. The old module accepted
`%Safetensors.FileTensor{}` and returned `Nx.Tensor`; keeping it would force this
file-format package to participate in every consumer's tensor-runtime version graph.

Use the raw APIs instead:

- `CrucibleSafetensors.Reader.read_tensor/2` for a whole tensor payload;
- `CrucibleSafetensors.Reader.read_slice/3` for a selected byte range;
- `CrucibleSafetensors.Reader.read_row_slice/4` or `ChunkReader.row_slices/3` for
  byte-aligned rank-2 row ranges;
- `CrucibleSafetensors.Manifest` for checkpoint inventory preflight;
- materialize returned binaries into Nx, Axon, Bumblebee, or another runtime in the
  consuming package rather than here.

This is a breaking 0.x change by design. The package now owns only SafeTensors file
format behavior and can coexist with any Nx/Bumblebee version selected by a caller.

## Original extraction notes

Source material for the Phase 2 implementation:

- `nshkrdotcom/trinity_coordinator` tag `v0.1.0-monolith`
- source commit `64144a2983950e5fc9f2db2d26323a576c7379a1`
- `lib/trinity_coordinator/sakana/safetensors_slice.ex`
- chunk-reader portions of `lib/trinity_coordinator/sakana/large_tensor_chunks.ex`

The initial implementation keeps the package independent of framework runtime
modules and owns only SafeTensors parsing, validation, slicing, checksums,
deterministic writing, and rank-2 row chunk helpers.

`Crucible.Safetensors.Slice` is a compatibility port of the legacy bounded
`%Safetensors.FileTensor{}` reader. It remains separate from the direct
binary reader/writer API so downstream code can migrate without changing tensor
materialization semantics in the same release window.
