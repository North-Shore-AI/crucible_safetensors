defmodule Crucible.Safetensors.VectorInspect do
  @moduledoc "Compatibility namespace for `CrucibleSafetensors.VectorInspect`."

  defdelegate inspect(path, opts \\ []), to: CrucibleSafetensors.VectorInspect
end
