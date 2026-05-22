defmodule Crucible.Safetensors.Writer do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.Writer`."

  defdelegate write(tensors, out_path, opts \\ []), to: CrucibleSafetensors.Writer
  defdelegate write!(tensors, out_path, opts \\ []), to: CrucibleSafetensors.Writer
end
