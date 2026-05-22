defmodule CrucibleSafetensors.LegacySliceTest do
  use ExUnit.Case, async: true

  alias Crucible.Safetensors.Slice, as: LegacySlice
  alias CrucibleSafetensors.Writer

  test "legacy row_slice reads lazy file tensor rows without materializing the whole tensor" do
    path = tmp_path("legacy_rows.safetensors")
    data = for value <- 1..8, into: <<>>, do: <<value::little-32>>
    Writer.write!(%{"rows" => %{dtype: :i32, shape: [4, 2], data: data}}, path)
    tensors = Safetensors.read!(path, lazy: true)
    lazy = Map.fetch!(tensors, "rows")

    assert %Nx.Tensor{shape: {2, 2}} = tensor = LegacySlice.row_slice!(lazy, 1, 2)
    assert Nx.to_flat_list(tensor) == [3, 4, 5, 6]
  end

  test "legacy materialize preserves binary backend tensors" do
    tensor = Nx.tensor([[1, 2]], type: :s32)
    assert %Nx.Tensor{shape: {1, 2}} = LegacySlice.materialize!(tensor)
  end

  defp tmp_path(name) do
    dir = Path.join(System.tmp_dir!(), "crucible_safetensors_tests")
    File.mkdir_p!(dir)
    Path.join(dir, "#{System.unique_integer([:positive])}_#{name}")
  end
end
