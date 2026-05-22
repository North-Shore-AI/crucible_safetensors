defmodule CrucibleSafetensors.ChunkReader do
  @moduledoc "Row-chunk helpers for large rank-2 SafeTensors payloads."

  alias CrucibleSafetensors.{Header, Reader, Slice, TensorInfo}

  @doc "Streams row slices for a rank-2 tensor."
  @spec row_slices(Header.t(), TensorInfo.t(), pos_integer()) :: Enumerable.t()
  def row_slices(%Header{} = header, %TensorInfo{shape: [rows, _cols]} = tensor, chunk_rows)
      when is_integer(chunk_rows) and chunk_rows > 0 do
    Stream.unfold(0, fn
      row_start when row_start >= rows ->
        nil

      row_start ->
        row_count = min(chunk_rows, rows - row_start)

        {%Slice{} = Reader.read_row_slice!(header, tensor, row_start, row_count),
         row_start + row_count}
    end)
  end

  @doc "Reads a single row chunk from a rank-2 tensor."
  @spec row_slice(Header.t(), TensorInfo.t(), non_neg_integer(), pos_integer()) ::
          {:ok, Slice.t()} | {:error, Exception.t()}
  def row_slice(%Header{} = header, %TensorInfo{} = tensor, row_start, row_count),
    do: Reader.read_row_slice(header, tensor, row_start, row_count)
end
