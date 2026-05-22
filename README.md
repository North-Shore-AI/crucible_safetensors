# CrucibleSafetensors

SafeTensors parsing, validation, bounded slicing, checksums, deterministic
writing, and row-chunk helpers for Elixir.

This package is intentionally narrow. It owns SafeTensors file-format behavior
and avoids provider, orchestration, inference, and tracing dependencies. New
callers should prefer `CrucibleSafetensors.Reader` and
`CrucibleSafetensors.Writer`; `Crucible.Safetensors.*` modules are compatibility
namespaces used during the TRINITY decomposition.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `crucible_safetensors` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_safetensors, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/crucible_safetensors>.

## CI

```sh
mix ci
```

Large fixture checks are opt-in:

```sh
TRINITY_LARGE_FIXTURE_DIR=~/p/g/n/trinity_coordinator/priv/sakana_trinity/adapted_qwen3_0_6b_layer26 \
  mix test --include large_safetensors
```
