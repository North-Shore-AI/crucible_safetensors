defmodule CrucibleSafetensors.Manifest do
  @moduledoc """
  Metadata-only tensor inventory and manifest validation.

  Manifest validation operates exclusively on the SafeTensors header. It is
  intended for checkpoint preflight before a tensor runtime maps names into a
  model graph.
  """

  alias CrucibleSafetensors.{Header, Reader, TensorInfo}

  @schema_version "crucible.safetensors.manifest.v1"

  @type expected_tensor :: %{
          optional(:dtype) => TensorInfo.dtype() | String.t(),
          optional(:shape) => [non_neg_integer()],
          optional(:nbytes) => non_neg_integer()
        }

  @doc "Returns a stable, name-sorted inventory of tensor metadata."
  @spec inventory(Header.t()) :: [map()]
  def inventory(%Header{tensors: tensors}) do
    tensors
    |> Map.values()
    |> Enum.sort_by(& &1.name)
    |> Enum.map(fn tensor ->
      %{
        name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        nbytes: tensor.nbytes,
        data_offsets: [tensor.data_start, tensor.data_end]
      }
    end)
  end

  @doc "Opens a file and validates its header against an expected tensor manifest."
  @spec validate_file(Path.t(), map(), keyword()) :: {:ok, map()} | {:error, term()}
  def validate_file(path, expected, opts \\ []) when is_binary(path) do
    with {:ok, header} <- Reader.open(path) do
      validate(header, expected, opts)
    end
  end

  @doc "Validates a parsed header against expected tensor names and metadata."
  @spec validate(Header.t(), map(), keyword()) :: {:ok, map()} | {:error, term()}
  def validate(header, expected, opts \\ [])

  def validate(%Header{} = header, expected, opts) when is_list(opts) do
    with {:ok, expected} <- normalize_expected(expected),
         {:ok, exact?} <- normalize_exact(opts) do
      actual_names = Map.keys(header.tensors) |> MapSet.new()
      expected_names = Map.keys(expected) |> MapSet.new()

      missing =
        expected_names |> MapSet.difference(actual_names) |> MapSet.to_list() |> Enum.sort()

      unexpected =
        actual_names |> MapSet.difference(expected_names) |> MapSet.to_list() |> Enum.sort()

      mismatches =
        expected
        |> Enum.flat_map(fn {name, spec} -> compare_tensor(header.tensors[name], name, spec) end)
        |> Enum.sort_by(fn mismatch -> {mismatch.name, mismatch.field} end)

      valid? = missing == [] and mismatches == [] and (not exact? or unexpected == [])

      {:ok,
       %{
         schema_version: @schema_version,
         path: header.path,
         file_size: header.file_size,
         tensor_count: map_size(header.tensors),
         expected_tensor_count: map_size(expected),
         exact?: exact?,
         valid?: valid?,
         missing: missing,
         unexpected: unexpected,
         mismatches: mismatches,
         inventory: inventory(header)
       }}
    end
  end

  def validate(_header, _expected, _opts), do: {:error, :invalid_manifest_arguments}

  defp normalize_expected(expected) when is_map(expected) do
    Enum.reduce_while(expected, {:ok, %{}}, fn
      {name, spec}, {:ok, acc} when is_binary(name) and name != "" and is_map(spec) ->
        case normalize_spec(spec) do
          {:ok, normalized} -> {:cont, {:ok, Map.put(acc, name, normalized)}}
          {:error, reason} -> {:halt, {:error, {:invalid_tensor_spec, name, reason}}}
        end

      {name, spec}, _acc ->
        {:halt, {:error, {:invalid_tensor_spec, name, spec}}}
    end)
  end

  defp normalize_expected(expected), do: {:error, {:invalid_manifest, expected}}

  defp normalize_spec(spec) do
    with {:ok, dtype} <- normalize_optional_dtype(value(spec, :dtype)),
         {:ok, shape} <- normalize_optional_shape(value(spec, :shape)),
         {:ok, nbytes} <- normalize_optional_nbytes(value(spec, :nbytes)) do
      {:ok, %{dtype: dtype, shape: shape, nbytes: nbytes}}
    end
  end

  defp normalize_optional_dtype(nil), do: {:ok, nil}

  defp normalize_optional_dtype(value) do
    case TensorInfo.normalize_dtype(value) do
      {:ok, dtype} -> {:ok, dtype}
      :error -> {:error, {:invalid_dtype, value}}
    end
  end

  defp normalize_optional_shape(nil), do: {:ok, nil}

  defp normalize_optional_shape(shape) when is_list(shape) do
    if Enum.all?(shape, &(is_integer(&1) and &1 >= 0)),
      do: {:ok, shape},
      else: {:error, {:invalid_shape, shape}}
  end

  defp normalize_optional_shape(shape), do: {:error, {:invalid_shape, shape}}

  defp normalize_optional_nbytes(nil), do: {:ok, nil}
  defp normalize_optional_nbytes(value) when is_integer(value) and value >= 0, do: {:ok, value}
  defp normalize_optional_nbytes(value), do: {:error, {:invalid_nbytes, value}}

  defp normalize_exact(opts) do
    case Keyword.get(opts, :exact, false) do
      value when is_boolean(value) -> {:ok, value}
      value -> {:error, {:invalid_exact, value}}
    end
  end

  defp compare_tensor(nil, _name, _spec), do: []

  defp compare_tensor(%TensorInfo{} = tensor, name, spec) do
    []
    |> maybe_mismatch(name, :dtype, spec.dtype, tensor.dtype)
    |> maybe_mismatch(name, :shape, spec.shape, tensor.shape)
    |> maybe_mismatch(name, :nbytes, spec.nbytes, tensor.nbytes)
  end

  defp maybe_mismatch(acc, _name, _field, nil, _actual), do: acc
  defp maybe_mismatch(acc, _name, _field, expected, expected), do: acc

  defp maybe_mismatch(acc, name, field, expected, actual) do
    [%{name: name, field: field, expected: expected, actual: actual} | acc]
  end

  defp value(spec, key), do: Map.get(spec, key, Map.get(spec, Atom.to_string(key)))
end
