defmodule CrucibleSafetensors.Reader do
  @moduledoc "SafeTensors reader with header validation and selective binary slices."

  alias CrucibleSafetensors.{Errors, Header, Slice, TensorInfo}

  @max_header_size 100_000_000

  @doc "Opens and validates a SafeTensors file without loading tensor payloads."
  @spec open(Path.t()) :: {:ok, Header.t()} | {:error, Exception.t()}
  def open(path) when is_binary(path) do
    {:ok, open!(path)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Opens and validates a SafeTensors file, raising on invalid input."
  @spec open!(Path.t()) :: Header.t()
  def open!(path) when is_binary(path) do
    path = Path.expand(path)
    file_size = file_size!(path)

    File.open!(path, [:read, :binary, :raw], fn file ->
      header_size = read_header_size!(file, path, file_size)
      header_json = read_exact!(file, 8, header_size, path)
      decoded = decode_header!(header_json, path)
      data_offset = 8 + header_size
      tensors = parse_tensors!(decoded, file_size - data_offset, path)
      metadata = metadata!(decoded, path)

      %Header{
        path: path,
        file_size: file_size,
        header_size: header_size,
        data_offset: data_offset,
        tensors: tensors,
        metadata: metadata
      }
    end)
  end

  @doc "Fetches a tensor metadata entry by name."
  @spec tensor(Header.t(), String.t()) :: {:ok, TensorInfo.t()} | {:error, :not_found}
  def tensor(%Header{tensors: tensors}, name) when is_binary(name) do
    case Map.fetch(tensors, name) do
      {:ok, tensor} -> {:ok, tensor}
      :error -> {:error, :not_found}
    end
  end

  @doc "Reads a byte slice relative to the start of the given tensor payload."
  @spec read_slice(Header.t(), TensorInfo.t(), Range.t() | {non_neg_integer(), non_neg_integer()}) ::
          {:ok, Slice.t()} | {:error, Exception.t()}
  def read_slice(%Header{} = header, %TensorInfo{} = tensor, byte_range) do
    {:ok, read_slice!(header, tensor, byte_range)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Reads a byte slice, raising on invalid ranges or short reads."
  @spec read_slice!(
          Header.t(),
          TensorInfo.t(),
          Range.t() | {non_neg_integer(), non_neg_integer()}
        ) ::
          Slice.t()
  def read_slice!(%Header{} = header, %TensorInfo{} = tensor, byte_range) do
    {offset, size} = normalize_byte_range!(byte_range)

    if offset + size > tensor.nbytes do
      raise Errors,
            "slice #{inspect(byte_range)} exceeds tensor #{inspect(tensor.name)} byte size #{tensor.nbytes}"
    end

    file_offset = header.data_offset + tensor.data_start + offset
    data = read_file_slice!(header.path, file_offset, size)

    %Slice{tensor: tensor, byte_offset: offset, byte_size: size, data: data}
  end

  @doc "Reads a whole tensor payload as a binary slice."
  @spec read_tensor(Header.t(), TensorInfo.t()) :: {:ok, Slice.t()} | {:error, Exception.t()}
  def read_tensor(%Header{} = header, %TensorInfo{} = tensor),
    do: read_slice(header, tensor, {0, tensor.nbytes})

  @doc "Reads a contiguous row slice from a rank-2 tensor."
  @spec read_row_slice(Header.t(), TensorInfo.t(), non_neg_integer(), pos_integer()) ::
          {:ok, Slice.t()} | {:error, Exception.t()}
  def read_row_slice(%Header{} = header, %TensorInfo{} = tensor, row_start, row_count) do
    {:ok, read_row_slice!(header, tensor, row_start, row_count)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Reads a contiguous row slice from a rank-2 tensor, raising on invalid input."
  @spec read_row_slice!(Header.t(), TensorInfo.t(), non_neg_integer(), pos_integer()) :: Slice.t()
  def read_row_slice!(
        %Header{} = header,
        %TensorInfo{shape: [rows, cols]} = tensor,
        row_start,
        row_count
      )
      when is_integer(row_start) and row_start >= 0 and is_integer(row_count) and row_count > 0 do
    if row_start + row_count > rows do
      raise Errors, "row slice #{inspect({row_start, row_count})} exceeds tensor rows #{rows}"
    end

    {:ok, bits} = TensorInfo.element_bits(tensor.dtype)
    row_bits = cols * bits

    if rem(row_bits, 8) != 0 do
      raise Errors,
            "row slice for #{inspect(tensor.name)} is not byte-aligned for dtype #{inspect(tensor.dtype)}"
    end

    row_bytes = div(row_bits, 8)
    read_slice!(header, tensor, {row_start * row_bytes, row_count * row_bytes})
  end

  def read_row_slice!(_header, %TensorInfo{name: name, shape: shape}, _row_start, _row_count) do
    raise Errors,
          "expected rank-2 tensor for row slice #{inspect(name)}, got shape #{inspect(shape)}"
  end

  defp file_size!(path) do
    case File.stat(path) do
      {:ok, %File.Stat{size: size}} -> size
      {:error, reason} -> raise File.Error, reason: reason, action: "stat", path: path
    end
  end

  defp read_header_size!(file, path, file_size) do
    case :file.pread(file, 0, 8) do
      {:ok, <<header_size::unsigned-little-64>>}
      when header_size <= @max_header_size and header_size <= file_size - 8 ->
        header_size

      {:ok, <<header_size::unsigned-little-64>>} when header_size > @max_header_size ->
        raise Errors,
              "header length #{header_size} exceeds #{@max_header_size} byte safety limit for #{path}"

      {:ok, <<header_size::unsigned-little-64>>} ->
        raise Errors, "header length #{header_size} exceeds file size for #{path}"

      {:ok, bytes} ->
        raise Errors,
              "short SafeTensors header length read from #{path}: #{byte_size(bytes)} bytes"

      {:error, reason} ->
        raise File.Error, reason: reason, action: "read", path: path
    end
  end

  defp read_exact!(file, offset, size, path) do
    case :file.pread(file, offset, size) do
      {:ok, binary} when byte_size(binary) == size ->
        binary

      {:ok, binary} ->
        raise Errors,
              "short SafeTensors read from #{path}: expected #{size} bytes, got #{byte_size(binary)}"

      {:error, reason} ->
        raise File.Error, reason: reason, action: "read", path: path
    end
  end

  defp decode_header!(<<?{, _rest::binary>> = json, path) do
    case Jason.decode(json, objects: :ordered_objects) do
      {:ok, %Jason.OrderedObject{} = decoded} ->
        ordered_object_to_map!(decoded, path, [])

      {:ok, _other} ->
        raise Errors, "SafeTensors header must be a JSON object in #{path}"

      {:error, error} ->
        raise Errors, "invalid SafeTensors header JSON in #{path}: #{Exception.message(error)}"
    end
  end

  defp decode_header!(_json, path) do
    raise Errors, "SafeTensors header must begin with a JSON object in #{path}"
  end

  defp ordered_object_to_map!(%Jason.OrderedObject{values: values}, path, json_path) do
    {decoded, _seen} =
      Enum.reduce(values, {%{}, MapSet.new()}, fn {key, value}, {acc, seen} ->
        if MapSet.member?(seen, key) do
          location = Enum.reverse([key | json_path]) |> Enum.join(".")
          raise Errors, "duplicate JSON object key #{inspect(location)} in #{path}"
        end

        decoded_value = ordered_value!(value, path, [key | json_path])
        {Map.put(acc, key, decoded_value), MapSet.put(seen, key)}
      end)

    decoded
  end

  defp ordered_value!(%Jason.OrderedObject{} = value, path, json_path),
    do: ordered_object_to_map!(value, path, json_path)

  defp ordered_value!(value, path, json_path) when is_list(value),
    do: Enum.map(value, &ordered_value!(&1, path, json_path))

  defp ordered_value!(value, _path, _json_path), do: value

  defp parse_tensors!(decoded, payload_size, path) do
    tensors =
      decoded
      |> Enum.reject(fn {name, _value} -> name == "__metadata__" end)
      |> Enum.map(fn {name, metadata} -> parse_tensor!(name, metadata, payload_size, path) end)

    validate_layout!(tensors, payload_size, path)
    Map.new(tensors, &{&1.name, &1})
  end

  defp parse_tensor!(
         name,
         %{"dtype" => dtype, "shape" => shape, "data_offsets" => [start, stop]},
         payload_size,
         path
       )
       when is_binary(name) and is_list(shape) and is_integer(start) and is_integer(stop) do
    dtype_atom = dtype!(dtype, path)
    validate_shape!(shape, name, path)

    unless start >= 0 and stop >= start and stop <= payload_size do
      raise Errors,
            "invalid data_offsets #{inspect([start, stop])} for tensor #{inspect(name)} in #{path}"
    end

    expected = expected_nbytes!(dtype_atom, shape, name, path)

    unless stop - start == expected do
      raise Errors,
            "tensor #{inspect(name)} declares #{stop - start} bytes but shape/dtype require #{expected}"
    end

    %TensorInfo{
      name: name,
      dtype: dtype_atom,
      shape: shape,
      data_start: start,
      data_end: stop,
      nbytes: stop - start
    }
  end

  defp parse_tensor!(name, metadata, _payload_size, path) do
    raise Errors, "invalid metadata for tensor #{inspect(name)} in #{path}: #{inspect(metadata)}"
  end

  defp dtype!(dtype, path) do
    case TensorInfo.normalize_dtype(dtype) do
      {:ok, normalized} -> normalized
      :error -> raise Errors, "unsupported dtype #{inspect(dtype)} in #{path}"
    end
  end

  defp metadata!(decoded, path) do
    metadata = Map.get(decoded, "__metadata__", %{})

    if is_map(metadata) and
         Enum.all?(metadata, fn {key, value} -> is_binary(key) and is_binary(value) end) do
      metadata
    else
      raise Errors, "SafeTensors __metadata__ must contain only string keys and values in #{path}"
    end
  end

  defp validate_shape!(shape, name, path) do
    unless Enum.all?(shape, &(is_integer(&1) and &1 >= 0)) do
      raise Errors, "invalid shape #{inspect(shape)} for tensor #{inspect(name)} in #{path}"
    end
  end

  defp validate_layout!(tensors, payload_size, path) do
    final_offset =
      tensors
      |> Enum.sort_by(& &1.data_start)
      |> Enum.reduce(0, fn tensor, expected_start ->
        if tensor.data_start != expected_start do
          raise Errors,
                "tensor #{inspect(tensor.name)} starts at #{tensor.data_start}, expected #{expected_start} in #{path}"
        end

        tensor.data_end
      end)

    if final_offset != payload_size do
      raise Errors,
            "tensor payload layout ends at #{final_offset} bytes but file contains #{payload_size} payload bytes in #{path}"
    end

    :ok
  end

  defp expected_nbytes!(dtype, shape, name, path) do
    case TensorInfo.payload_nbytes(dtype, shape) do
      {:ok, nbytes} ->
        nbytes

      {:error, :misaligned} ->
        raise Errors,
              "tensor #{inspect(name)} shape/dtype bit length is not byte-aligned in #{path}"

      {:error, reason} ->
        raise Errors,
              "cannot determine payload size for tensor #{inspect(name)} in #{path}: #{inspect(reason)}"
    end
  end

  defp normalize_byte_range!({offset, size})
       when is_integer(offset) and offset >= 0 and is_integer(size) and size >= 0,
       do: {offset, size}

  defp normalize_byte_range!(first..last//1 = range) when first >= 0 and last >= first,
    do: {first, Range.size(range)}

  defp normalize_byte_range!(range), do: raise(Errors, "invalid byte range #{inspect(range)}")

  defp read_file_slice!(path, offset, size) do
    File.open!(path, [:read, :binary, :raw], fn file ->
      read_exact!(file, offset, size, path)
    end)
  end
end
