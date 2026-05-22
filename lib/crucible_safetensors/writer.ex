defmodule CrucibleSafetensors.Writer do
  @moduledoc "Deterministic SafeTensors writer for binary tensor payloads."

  alias CrucibleSafetensors.Errors

  @dtype_names %{
    f16: "F16",
    bf16: "BF16",
    f32: "F32",
    i32: "I32",
    i64: "I64"
  }

  @dtype_bytes %{f16: 2, bf16: 2, f32: 4, i32: 4, i64: 8}

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
          "dtype" => Map.fetch!(@dtype_names, dtype),
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

    expected = Enum.product(shape) * Map.fetch!(@dtype_bytes, dtype)

    unless is_binary(data) and byte_size(data) == expected do
      raise Errors,
            "tensor #{inspect(name)} payload has #{byte_size(data)} bytes but shape/dtype require #{expected}"
    end

    {dtype, shape, data}
  end

  defp normalize_tensor!(name, tensor) do
    raise Errors, "invalid tensor payload for #{inspect(name)}: #{inspect(tensor)}"
  end

  defp normalize_dtype!(dtype) when is_atom(dtype) and is_map_key(@dtype_names, dtype), do: dtype

  defp normalize_dtype!(dtype) when is_binary(dtype) do
    case Enum.find(@dtype_names, fn {_atom, name} -> name == dtype end) do
      {atom, _name} -> atom
      nil -> raise Errors, "unsupported dtype #{inspect(dtype)}"
    end
  end

  defp normalize_dtype!(dtype), do: raise(Errors, "unsupported dtype #{inspect(dtype)}")

  defp encode_header!(entries, metadata) do
    entries =
      if metadata == %{} do
        entries
      else
        [{"__metadata__", metadata} | entries]
      end

    encode_json_object(entries)
  end

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
