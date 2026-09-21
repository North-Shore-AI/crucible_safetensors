defmodule CrucibleSafetensors.TensorInfo do
  @moduledoc """
  Validated metadata for one tensor inside a SafeTensors file.

  The module also owns the package's SafeTensors dtype vocabulary so readers,
  writers, inspectors, and manifest validation agree on wire names and bit
  widths without depending on Nx or another tensor runtime.
  """

  @dtype_info %{
    bool: {"BOOL", 8},
    f4: {"F4", 4},
    f6_e2m3: {"F6_E2M3", 6},
    f6_e3m2: {"F6_E3M2", 6},
    u8: {"U8", 8},
    i8: {"I8", 8},
    f8_e5m2: {"F8_E5M2", 8},
    f8_e4m3: {"F8_E4M3", 8},
    f8_e8m0: {"F8_E8M0", 8},
    f8_e4m3fnuz: {"F8_E4M3FNUZ", 8},
    f8_e5m2fnuz: {"F8_E5M2FNUZ", 8},
    i16: {"I16", 16},
    u16: {"U16", 16},
    f16: {"F16", 16},
    bf16: {"BF16", 16},
    i32: {"I32", 32},
    u32: {"U32", 32},
    f32: {"F32", 32},
    c64: {"C64", 64},
    f64: {"F64", 64},
    i64: {"I64", 64},
    u64: {"U64", 64}
  }

  @wire_to_dtype Map.new(@dtype_info, fn {dtype, {wire, _bits}} -> {wire, dtype} end)

  @enforce_keys [:name, :dtype, :shape, :data_start, :data_end, :nbytes]
  defstruct [:name, :dtype, :shape, :data_start, :data_end, :nbytes]

  @type dtype ::
          :bool
          | :f4
          | :f6_e2m3
          | :f6_e3m2
          | :u8
          | :i8
          | :f8_e5m2
          | :f8_e4m3
          | :f8_e8m0
          | :f8_e4m3fnuz
          | :f8_e5m2fnuz
          | :i16
          | :u16
          | :f16
          | :bf16
          | :i32
          | :u32
          | :f32
          | :c64
          | :f64
          | :i64
          | :u64

  @type t :: %__MODULE__{
          name: String.t(),
          dtype: dtype(),
          shape: [non_neg_integer()],
          data_start: non_neg_integer(),
          data_end: non_neg_integer(),
          nbytes: non_neg_integer()
        }

  @doc "Returns all dtypes understood by the raw SafeTensors layer."
  @spec supported_dtypes() :: [dtype()]
  def supported_dtypes, do: Map.keys(@dtype_info) |> Enum.sort()

  @doc "Normalizes an atom or SafeTensors dtype string."
  @spec normalize_dtype(dtype() | String.t()) :: {:ok, dtype()} | :error
  def normalize_dtype(dtype) when is_atom(dtype) do
    if is_map_key(@dtype_info, dtype), do: {:ok, dtype}, else: :error
  end

  def normalize_dtype(dtype) when is_binary(dtype) do
    normalized = String.upcase(dtype)

    case Map.fetch(@wire_to_dtype, normalized) do
      {:ok, value} -> {:ok, value}
      :error -> :error
    end
  end

  def normalize_dtype(_dtype), do: :error

  @doc "Returns the canonical SafeTensors wire name for a dtype."
  @spec wire_dtype(dtype() | String.t()) :: {:ok, String.t()} | :error
  def wire_dtype(dtype) do
    with {:ok, normalized} <- normalize_dtype(dtype),
         {wire, _bits} <- Map.fetch!(@dtype_info, normalized) do
      {:ok, wire}
    end
  end

  @doc "Returns the number of bits used by one element of the dtype."
  @spec element_bits(dtype() | String.t()) :: {:ok, pos_integer()} | :error
  def element_bits(dtype) do
    with {:ok, normalized} <- normalize_dtype(dtype),
         {_wire, bits} <- Map.fetch!(@dtype_info, normalized) do
      {:ok, bits}
    end
  end

  @doc "Returns the encoded payload byte size for a dtype and shape."
  @spec payload_nbytes(dtype() | String.t(), [non_neg_integer()]) ::
          {:ok, non_neg_integer()} | {:error, :invalid_shape | :unsupported_dtype | :misaligned}
  def payload_nbytes(dtype, shape) when is_list(shape) do
    with true <- valid_shape?(shape),
         {:ok, bits} <- element_bits(dtype) do
      shape
      |> Enum.product()
      |> Kernel.*(bits)
      |> aligned_payload_nbytes()
    else
      false -> {:error, :invalid_shape}
      :error -> {:error, :unsupported_dtype}
    end
  end

  def payload_nbytes(_dtype, _shape), do: {:error, :invalid_shape}

  defp valid_shape?(shape),
    do: Enum.all?(shape, &(is_integer(&1) and &1 >= 0))

  defp aligned_payload_nbytes(total_bits) when rem(total_bits, 8) == 0,
    do: {:ok, div(total_bits, 8)}

  defp aligned_payload_nbytes(_total_bits), do: {:error, :misaligned}
end
