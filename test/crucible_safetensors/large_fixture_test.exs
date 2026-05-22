defmodule CrucibleSafetensors.LargeFixtureTest do
  use ExUnit.Case, async: false

  alias CrucibleSafetensors.{ChunkReader, Reader}

  @tag :large_safetensors
  test "large fixture row chunks concatenate to the same bytes as a full read" do
    fixture_dir =
      System.get_env(
        "TRINITY_LARGE_FIXTURE_DIR",
        Path.expand("~/p/g/n/trinity_coordinator/priv/sakana_trinity/adapted_qwen3_0_6b_layer26")
      )

    path =
      fixture_dir
      |> Path.join("router_head.safetensors")
      |> Path.expand()

    assert File.regular?(path)

    header = Reader.open!(path)
    tensor = pick_rank2_tensor!(header)
    {:ok, whole} = Reader.read_tensor(header, tensor)

    chunked =
      header
      |> ChunkReader.row_slices(tensor, 1)
      |> Enum.map_join(& &1.data)

    assert chunked == whole.data
  end

  defp pick_rank2_tensor!(header) do
    header.tensors
    |> Map.values()
    |> Enum.find(fn tensor -> match?([rows, _cols] when rows > 1, tensor.shape) end) ||
      flunk("expected at least one rank-2 tensor in #{header.path}")
  end
end
