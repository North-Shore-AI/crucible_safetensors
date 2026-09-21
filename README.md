<p align="center">
  <img src="assets/crucible_safetensors.svg" alt="CrucibleSafetensors Logo" width="200px" />
</p>

# CrucibleSafetensors

<p align="center">
  <a href="https://github.com/North-Shore-AI/crucible_safetensors/actions/workflows/ci.yml">
    <img src="https://github.com/North-Shore-AI/crucible_safetensors/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status" />
  </a>
  <a href="https://github.com/North-Shore-AI/crucible_safetensors/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/North-Shore-AI/crucible_safetensors" alt="GitHub License" />
  </a>
</p>

SafeTensors parsing, validation, selective range reads, streaming checksums,
manifest inspection, deterministic writing, and row-chunk helpers for Elixir.

This package is intentionally narrow. It owns SafeTensors file-format behavior
and avoids provider, orchestration, inference, and tracing dependencies. New
callers should prefer `CrucibleSafetensors.Reader` and
`CrucibleSafetensors.Writer`. Compatibility namespaces for the generic file-format
APIs remain available under `Crucible.Safetensors.*`; the old Nx/FileTensor bridge
was removed in 0.2.0 so this package has no tensor-runtime dependency.

## What It Provides

- `CrucibleSafetensors.Reader` opens a `.safetensors` file, validates the
  header, and reads selected tensor byte ranges without materializing the whole
  file.
- `CrucibleSafetensors.Writer` writes deterministic `.safetensors` files from
  binary tensor payloads.
- `CrucibleSafetensors.ChunkReader` streams rank-2 row chunks for large tensors.
- `CrucibleSafetensors.Checksum` incrementally computes and verifies SHA-256 file digests.
- `CrucibleSafetensors.Manifest` inventories and validates expected tensor names,
  dtypes, shapes, and byte sizes using header metadata only.
- `CrucibleSafetensors.VectorInspect` validates one selected tensor key, dtype,
  shape, element count, and file digest without loading its payload.

The package does not own model loading, artifact fetching, provider calls,
runtime orchestration, tracing, or application configuration.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `crucible_safetensors` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_safetensors, "~> 0.2.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/crucible_safetensors>.

## Reading

```elixir
alias CrucibleSafetensors.Reader

header = Reader.open!("model.safetensors")
{:ok, tensor} = Reader.tensor(header, "decoder.weight")
{:ok, slice} = Reader.read_tensor(header, tensor)

slice.tensor.name
slice.byte_size
slice.data
```

Read one byte range:

```elixir
{:ok, slice} = Reader.read_slice(header, tensor, {0, 4096})
```

Read row slices from a rank-2 tensor:

```elixir
{:ok, row_slice} = Reader.read_row_slice(header, tensor, 0, 16)
```

## Writing

```elixir
alias CrucibleSafetensors.Writer

Writer.write!(
  %{
    "weights" => %{
      dtype: :f32,
      shape: [2, 2],
      data: <<0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64>>
    }
  },
  "out/weights.safetensors",
  metadata: %{"source" => "example"}
)
```

Supported writer dtypes are `:f16`, `:bf16`, `:f32`, `:i32`, and `:i64`.
Writer output is sorted by tensor name so repeated writes are byte-stable.

## Row Chunking

`CrucibleSafetensors.ChunkReader.row_slices/3` returns a stream of rank-2 row slices:

```elixir
header
|> CrucibleSafetensors.ChunkReader.row_slices(tensor, 512)
|> Enum.each(fn slice ->
  IO.iodata_length(slice.data)
end)
```

This is useful for validating or copying large matrix tensors without reading
the entire payload at once.

## Checksums

```elixir
{:ok, sha256} = CrucibleSafetensors.Checksum.file_sha256("out/weights.safetensors")
```

The checksum helper reads incrementally and returns lowercase hexadecimal SHA-256 text.
It also accepts `sha256:<hex>` values for explicit artifact verification:

```elixir
:ok = case CrucibleSafetensors.Checksum.verify_file("out/weights.safetensors", expected) do
  {:ok, _digest} -> :ok
  {:error, reason} -> raise inspect(reason)
end
```

## Manifest Validation

```elixir
alias CrucibleSafetensors.Manifest

{:ok, report} =
  Manifest.validate_file("model.safetensors", %{
    "encoder.weight" => %{dtype: :bf16, shape: [1024, 1024]},
    "classifier.bias" => %{dtype: :f32, shape: [3]}
  }, exact: false)

report.valid?
report.missing
report.mismatches
```

Manifest validation reads only the SafeTensors header. It is intended for model
checkpoint preflight before a runtime imports tensor bytes.

## Vector Inspection

```elixir
{:ok, report} =
  CrucibleSafetensors.VectorInspect.inspect("router_vector.safetensors",
    tensor_key: "router_vector",
    expected_count: 19_456
  )
```

The report is metadata-only and does not materialize the tensor payload.

## CI

```sh
mix ci
```

Large fixture checks are opt-in:

```sh
mkdir -p tmp
ln -s /path/to/safetensors_bundle tmp/crucible_safetensors_fixture
mix test --include large_safetensors
```

`mix ci` runs dependency fetch, format check, warning-as-error compile, tests,
Credo strict, Dialyzer, and docs generation.
