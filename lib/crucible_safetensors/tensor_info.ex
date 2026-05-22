defmodule CrucibleSafetensors.TensorInfo do
  @moduledoc "Validated metadata for one tensor inside a SafeTensors file."

  @enforce_keys [:name, :dtype, :shape, :data_start, :data_end, :nbytes]
  defstruct [:name, :dtype, :shape, :data_start, :data_end, :nbytes]

  @type dtype :: :f16 | :bf16 | :f32 | :i32 | :i64

  @type t :: %__MODULE__{
          name: String.t(),
          dtype: dtype(),
          shape: [non_neg_integer()],
          data_start: non_neg_integer(),
          data_end: non_neg_integer(),
          nbytes: non_neg_integer()
        }
end
