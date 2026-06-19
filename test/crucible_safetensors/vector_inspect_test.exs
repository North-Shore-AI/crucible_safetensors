defmodule CrucibleSafetensors.VectorInspectTest do
  use ExUnit.Case, async: true

  alias CrucibleSafetensors.{VectorInspect, Writer}

  test "reports valid tensor key shape dtype count and digest" do
    path = tmp_path("valid_vector.safetensors")

    Writer.write!(
      %{
        "router_vector" => %{
          dtype: :f32,
          shape: [3],
          data: <<0::little-32, 1::little-32, 2::little-32>>
        }
      },
      path
    )

    assert {:ok, report} =
             VectorInspect.inspect(path,
               tensor_key: "router_vector",
               expected_shape: [3],
               expected_dtype: :f32,
               expected_count: 3
             )

    assert report.valid?
    assert report.status == :ok
    assert report.tensor_keys == ["router_vector"]
    assert report.tensor_key == "router_vector"
    assert report.dtype == :f32
    assert report.shape == [3]
    assert report.element_count == 3
    assert report.file_sha256 =~ "sha256:"
  end

  test "reports missing tensor key without loading payload" do
    path = tmp_path("missing_key.safetensors")
    Writer.write!(%{"other" => %{dtype: :i32, shape: [1], data: <<1::little-32>>}}, path)

    assert {:ok, report} = VectorInspect.inspect(path, tensor_key: "router_vector")

    refute report.valid?
    assert report.status == :missing_tensor_key
    assert report.tensor_keys == ["other"]
  end

  test "reports shape mismatch" do
    path = tmp_path("shape_mismatch.safetensors")

    Writer.write!(
      %{"router_vector" => %{dtype: :f32, shape: [2], data: <<0::little-32, 1::little-32>>}},
      path
    )

    assert {:ok, report} =
             VectorInspect.inspect(path, tensor_key: "router_vector", expected_shape: [3])

    refute report.valid?
    assert report.status == :shape_mismatch
    assert report.expected_shape == [3]
  end

  test "reports dtype mismatch" do
    path = tmp_path("dtype_mismatch.safetensors")
    Writer.write!(%{"router_vector" => %{dtype: :i32, shape: [1], data: <<1::little-32>>}}, path)

    assert {:ok, report} =
             VectorInspect.inspect(path, tensor_key: "router_vector", expected_dtype: :f32)

    refute report.valid?
    assert report.status == :dtype_mismatch
    assert report.expected_dtype == :f32
  end

  test "reports count mismatch independently from shape" do
    path = tmp_path("count_mismatch.safetensors")

    Writer.write!(
      %{"router_vector" => %{dtype: :f32, shape: [2, 2], data: <<0::size(128)>>}},
      path
    )

    assert {:ok, report} =
             VectorInspect.inspect(path, tensor_key: "router_vector", expected_count: 5)

    refute report.valid?
    assert report.status == :element_count_mismatch
    assert report.element_count == 4
  end

  defp tmp_path(name) do
    dir = Path.join(["tmp", "test", "vector_inspect"])
    File.mkdir_p!(dir)
    Path.join(dir, "#{System.unique_integer([:positive])}_#{name}")
  end
end
