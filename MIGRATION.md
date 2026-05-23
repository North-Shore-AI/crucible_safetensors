# Migration Notes

This repo was scaffolded during the monolith extraction so framework consumers
could resolve local path dependencies before the public GitHub repos existed.

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
