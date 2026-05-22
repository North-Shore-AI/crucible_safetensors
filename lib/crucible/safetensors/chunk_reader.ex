defmodule Crucible.Safetensors.ChunkReader do
  @moduledoc "Compatibility namespace for row-chunk SafeTensors helpers."

  alias CrucibleSafetensors.{Header, Slice, TensorInfo}

  @doc "Streams row slices for a rank-2 tensor."
  @spec row_slices(Header.t(), TensorInfo.t(), pos_integer()) :: Enumerable.t()
  defdelegate row_slices(header, tensor, chunk_rows), to: CrucibleSafetensors.ChunkReader

  @doc "Reads a single row chunk from a rank-2 tensor."
  @spec row_slice(Header.t(), TensorInfo.t(), non_neg_integer(), pos_integer()) ::
          {:ok, Slice.t()} | {:error, Exception.t()}
  defdelegate row_slice(header, tensor, row_start, row_count),
    to: CrucibleSafetensors.ChunkReader
end
