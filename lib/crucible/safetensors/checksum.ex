defmodule Crucible.Safetensors.Checksum do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.Checksum`."

  defdelegate file_sha256(path, opts \\ []), to: CrucibleSafetensors.Checksum
  defdelegate file_sha256!(path, opts \\ []), to: CrucibleSafetensors.Checksum
  defdelegate verify_file(path, expected, opts \\ []), to: CrucibleSafetensors.Checksum
  defdelegate verify_file!(path, expected, opts \\ []), to: CrucibleSafetensors.Checksum
end
