defmodule Crucible.Safetensors.Checksum do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.Checksum`."

  defdelegate file_sha256(path), to: CrucibleSafetensors.Checksum
  defdelegate file_sha256!(path), to: CrucibleSafetensors.Checksum
end
