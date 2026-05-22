defmodule CrucibleSafetensors.Slice do
  @moduledoc "A bounded binary slice read from one SafeTensors tensor."

  alias CrucibleSafetensors.TensorInfo

  @enforce_keys [:tensor, :byte_offset, :byte_size, :data]
  defstruct [:tensor, :byte_offset, :byte_size, :data]

  @type t :: %__MODULE__{
          tensor: TensorInfo.t(),
          byte_offset: non_neg_integer(),
          byte_size: non_neg_integer(),
          data: binary()
        }
end
