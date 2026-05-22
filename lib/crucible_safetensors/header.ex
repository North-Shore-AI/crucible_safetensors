defmodule CrucibleSafetensors.Header do
  @moduledoc "Validated SafeTensors header and file layout."

  alias CrucibleSafetensors.TensorInfo

  @enforce_keys [:path, :file_size, :header_size, :data_offset, :tensors, :metadata]
  defstruct [:path, :file_size, :header_size, :data_offset, :tensors, :metadata]

  @type t :: %__MODULE__{
          path: Path.t(),
          file_size: non_neg_integer(),
          header_size: non_neg_integer(),
          data_offset: non_neg_integer(),
          tensors: %{String.t() => TensorInfo.t()},
          metadata: map()
        }
end
