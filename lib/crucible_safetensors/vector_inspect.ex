defmodule CrucibleSafetensors.VectorInspect do
  @moduledoc """
  Bounded SafeTensors vector inspection.

  This module validates tensor metadata and file identity without materializing
  tensor payload bytes. It is intentionally generic and contains no
  framework-specific vector lengths or artifact names.
  """

  alias CrucibleSafetensors.{Checksum, Reader}

  @schema_version "crucible.safetensors.vector_inspect.v1"
  @dtype_aliases %{
    :f16 => :f16,
    "f16" => :f16,
    "F16" => :f16,
    :bf16 => :bf16,
    "bf16" => :bf16,
    "BF16" => :bf16,
    :f32 => :f32,
    "f32" => :f32,
    "F32" => :f32,
    :i32 => :i32,
    "i32" => :i32,
    "I32" => :i32,
    :i64 => :i64,
    "i64" => :i64,
    "I64" => :i64
  }

  @spec inspect(Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def inspect(path, opts \\ [])

  def inspect(path, opts) when is_binary(path) and is_list(opts) do
    tensor_key = Keyword.get(opts, :tensor_key)

    with {:ok, header} <- Reader.open(path),
         {:ok, sha256} <- Checksum.file_sha256(path),
         {:ok, expected_dtype} <- expected_dtype(opts),
         {:ok, expected_shape} <- expected_shape(opts),
         {:ok, expected_count} <- expected_count(opts) do
      tensor_keys = header.tensors |> Map.keys() |> Enum.sort()

      report =
        base_report(path, sha256, tensor_keys, tensor_key)
        |> add_tensor(header, tensor_key)
        |> add_expectations(expected_dtype, expected_shape, expected_count)
        |> validate()

      {:ok, report}
    end
  end

  def inspect(path, _opts), do: {:error, {:invalid_safetensors_path, path}}

  defp base_report(path, sha256, tensor_keys, tensor_key) do
    %{
      schema_version: @schema_version,
      path: Path.expand(path),
      file_sha256: "sha256:" <> sha256,
      tensor_keys: tensor_keys,
      tensor_key: tensor_key,
      tensor_present?: false,
      dtype: nil,
      shape: nil,
      element_count: nil,
      nbytes: nil,
      expected_dtype: nil,
      expected_shape: nil,
      expected_count: nil,
      valid?: false,
      status: :missing_tensor_key,
      status_reason: nil
    }
  end

  defp add_tensor(report, _header, nil) do
    %{report | status: :missing_tensor_key, status_reason: "tensor_key option is required"}
  end

  defp add_tensor(report, header, tensor_key) when is_binary(tensor_key) do
    case Reader.tensor(header, tensor_key) do
      {:ok, tensor} ->
        %{
          report
          | tensor_present?: true,
            dtype: tensor.dtype,
            shape: tensor.shape,
            element_count: Enum.product(tensor.shape),
            nbytes: tensor.nbytes,
            status: :ok
        }

      {:error, :not_found} ->
        %{report | status: :missing_tensor_key, status_reason: "selected tensor key not found"}
    end
  end

  defp add_tensor(report, _header, _tensor_key) do
    %{report | status: :missing_tensor_key, status_reason: "tensor_key must be a string"}
  end

  defp add_expectations(report, expected_dtype, expected_shape, expected_count) do
    %{
      report
      | expected_dtype: expected_dtype,
        expected_shape: expected_shape,
        expected_count: expected_count
    }
  end

  defp validate(%{tensor_present?: false} = report), do: report

  defp validate(%{dtype: dtype, expected_dtype: expected} = report)
       when not is_nil(expected) and dtype != expected do
    %{report | valid?: false, status: :dtype_mismatch, status_reason: "dtype does not match"}
  end

  defp validate(%{shape: shape, expected_shape: expected} = report)
       when not is_nil(expected) and shape != expected do
    %{report | valid?: false, status: :shape_mismatch, status_reason: "shape does not match"}
  end

  defp validate(%{element_count: count, expected_count: expected} = report)
       when not is_nil(expected) and count != expected do
    %{
      report
      | valid?: false,
        status: :element_count_mismatch,
        status_reason: "element count does not match"
    }
  end

  defp validate(report), do: %{report | valid?: true, status: :ok, status_reason: nil}

  defp expected_dtype(opts) do
    case Keyword.get(opts, :expected_dtype) do
      nil ->
        {:ok, nil}

      value ->
        case Map.fetch(@dtype_aliases, value) do
          {:ok, dtype} -> {:ok, dtype}
          :error -> {:error, {:invalid_expected_dtype, value}}
        end
    end
  end

  defp expected_shape(opts) do
    case Keyword.get(opts, :expected_shape) do
      nil ->
        {:ok, nil}

      shape when is_list(shape) ->
        if Enum.all?(shape, &(is_integer(&1) and &1 >= 0)),
          do: {:ok, shape},
          else: {:error, {:invalid_expected_shape, shape}}

      other ->
        {:error, {:invalid_expected_shape, other}}
    end
  end

  defp expected_count(opts) do
    case Keyword.get(opts, :expected_count) do
      nil -> {:ok, nil}
      value when is_integer(value) and value >= 0 -> {:ok, value}
      other -> {:error, {:invalid_expected_count, other}}
    end
  end
end
