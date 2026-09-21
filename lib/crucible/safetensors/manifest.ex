defmodule Crucible.Safetensors.Manifest do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.Manifest`."

  defdelegate inventory(header), to: CrucibleSafetensors.Manifest
  defdelegate validate(header, expected, opts \\ []), to: CrucibleSafetensors.Manifest
  defdelegate validate_file(path, expected, opts \\ []), to: CrucibleSafetensors.Manifest
end
