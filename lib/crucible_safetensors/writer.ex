defmodule CrucibleSafetensors.Writer do
  @moduledoc "Deterministic SafeTensors writer for binary tensor payloads."

  alias CrucibleSafetensors.{Errors, TensorInfo}

  @type tensor_payload :: %{
          required(:dtype) => atom() | String.t(),
          required(:shape) => [non_neg_integer()],
          required(:data) => binary()
        }

  @doc "Writes a SafeTensors file from a map of tensor names to binary payload metadata."
  @spec write(%{String.t() => tensor_payload()}, Path.t(), keyword()) ::
          {:ok, Path.t()} | {:error, Exception.t()}
  def write(tensors, out_path, opts \\ []) when is_map(tensors) and is_binary(out_path) do
    {:ok, write!(tensors, out_path, opts)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Writes a SafeTensors file, raising on invalid payloads."
  @spec write!(%{String.t() => tensor_payload()}, Path.t(), keyword()) :: Path.t()
  def write!(tensors, out_path, opts \\ []) when is_map(tensors) and is_binary(out_path) do
    metadata = Keyword.get(opts, :metadata, %{})
    {entries, payload} = build_entries_and_payload!(tensors)
    header = encode_header!(entries, metadata)

    File.mkdir_p!(Path.dirname(out_path))
    File.write!(out_path, [<<byte_size(header)::unsigned-little-64>>, header, payload])
    out_path
  end

  defp build_entries_and_payload!(tensors) do
    tensors
    |> Enum.sort_by(fn {name, _tensor} -> name end)
    |> Enum.reduce({[], [], 0}, fn {name, tensor}, {entries, payloads, offset} ->
      {dtype, shape, data} = normalize_tensor!(name, tensor)
      next_offset = offset + byte_size(data)

      entry = {
        name,
        %{
          "dtype" => wire_dtype!(dtype),
          "shape" => shape,
          "data_offsets" => [offset, next_offset]
        }
      }

      {[entry | entries], [data | payloads], next_offset}
    end)
    |> then(fn {entries, payloads, _offset} ->
      {Enum.reverse(entries), Enum.reverse(payloads)}
    end)
  end

  defp normalize_tensor!(name, %{dtype: dtype, shape: shape, data: data}) do
    dtype = normalize_dtype!(dtype)

    unless is_list(shape) and Enum.all?(shape, &(is_integer(&1) and &1 >= 0)) do
      raise Errors, "invalid shape for tensor #{inspect(name)}: #{inspect(shape)}"
    end

    expected = payload_nbytes!(dtype, shape, name)

    unless is_binary(data) and byte_size(data) == expected do
      raise Errors,
            "tensor #{inspect(name)} payload has #{byte_size(data)} bytes but shape/dtype require #{expected}"
    end

    {dtype, shape, data}
  end

  defp normalize_tensor!(name, tensor) do
    raise Errors, "invalid tensor payload for #{inspect(name)}: #{inspect(tensor)}"
  end

  defp payload_nbytes!(dtype, shape, name) do
    case TensorInfo.payload_nbytes(dtype, shape) do
      {:ok, nbytes} ->
        nbytes

      {:error, :misaligned} ->
        raise Errors, "tensor #{inspect(name)} shape/dtype bit length is not byte-aligned"

      {:error, reason} ->
        raise Errors, "invalid tensor #{inspect(name)} shape/dtype: #{inspect(reason)}"
    end
  end

  defp normalize_dtype!(dtype) do
    case TensorInfo.normalize_dtype(dtype) do
      {:ok, normalized} -> normalized
      :error -> raise Errors, "unsupported dtype #{inspect(dtype)}"
    end
  end

  defp wire_dtype!(dtype) do
    {:ok, wire} = TensorInfo.wire_dtype(dtype)
    wire
  end

  defp encode_header!(entries, metadata) do
    metadata = normalize_metadata!(metadata)

    entries =
      if metadata == %{} do
        entries
      else
        [{"__metadata__", metadata} | entries]
      end

    encode_json_object(entries)
  end

  defp normalize_metadata!(metadata) when is_map(metadata) do
    if Enum.all?(metadata, fn {key, value} -> is_binary(key) and is_binary(value) end) do
      metadata
    else
      raise Errors, "metadata must contain only string keys and values"
    end
  end

  defp normalize_metadata!(metadata),
    do: raise(Errors, "metadata must be a map, got: #{inspect(metadata)}")

  defp encode_json_object(entries) do
    body =
      entries
      |> Enum.map(fn {key, value} ->
        [Jason.encode!(key), ?:, Jason.encode!(value)]
      end)
      |> Enum.intersperse(?,)

    IO.iodata_to_binary([?{, body, ?}])
  end
end
