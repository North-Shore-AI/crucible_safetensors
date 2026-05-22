defmodule Crucible.Safetensors.Reader do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.Reader`."

  defdelegate open(path), to: CrucibleSafetensors.Reader
  defdelegate open!(path), to: CrucibleSafetensors.Reader
  defdelegate tensor(header, name), to: CrucibleSafetensors.Reader
  defdelegate read_slice(header, tensor, byte_range), to: CrucibleSafetensors.Reader
  defdelegate read_slice!(header, tensor, byte_range), to: CrucibleSafetensors.Reader
  defdelegate read_tensor(header, tensor), to: CrucibleSafetensors.Reader
  defdelegate read_row_slice(header, tensor, row_start, row_count), to: CrucibleSafetensors.Reader

  defdelegate read_row_slice!(header, tensor, row_start, row_count),
    to: CrucibleSafetensors.Reader
end
