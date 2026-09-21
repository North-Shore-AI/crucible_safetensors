defmodule CrucibleSafetensors.ManifestTest do
  use ExUnit.Case, async: true

  alias CrucibleSafetensors.{Manifest, Reader, Writer}

  test "validates required tensor names, dtypes, shapes, and byte sizes from the header" do
    path = fixture_file()
    header = Reader.open!(path)

    expected = %{
      "encoder.weight" => %{dtype: "BF16", shape: [2, 2], nbytes: 8},
      "head.bias" => %{dtype: :f32, shape: [2]}
    }

    assert {:ok, report} = Manifest.validate(header, expected, exact: true)
    assert report.valid?
    assert report.missing == []
    assert report.unexpected == []
    assert report.mismatches == []
    assert Enum.map(report.inventory, & &1.name) == ["encoder.weight", "head.bias"]
  end

  test "reports missing, unexpected, and metadata mismatches without reading payloads" do
    path = fixture_file()

    expected = %{
      "encoder.weight" => %{dtype: :f32, shape: [4, 1], nbytes: 16},
      "missing.weight" => %{}
    }

    assert {:ok, report} = Manifest.validate_file(path, expected, exact: true)
    refute report.valid?
    assert report.missing == ["missing.weight"]
    assert report.unexpected == ["head.bias"]

    assert %{name: "encoder.weight", field: :dtype, expected: :f32, actual: :bf16} in report.mismatches

    assert %{name: "encoder.weight", field: :shape, expected: [4, 1], actual: [2, 2]} in report.mismatches

    assert %{name: "encoder.weight", field: :nbytes, expected: 16, actual: 8} in report.mismatches
  end

  test "allows additional tensors unless exact matching is requested" do
    path = fixture_file()

    assert {:ok, report} =
             Manifest.validate_file(path, %{"encoder.weight" => %{dtype: :bf16}})

    assert report.valid?
    assert report.unexpected == ["head.bias"]
  end

  test "rejects malformed expected manifests" do
    path = fixture_file()
    header = Reader.open!(path)

    assert {:error, {:invalid_tensor_spec, "encoder.weight", {:invalid_dtype, :wat}}} =
             Manifest.validate(header, %{"encoder.weight" => %{dtype: :wat}})

    assert {:error, {:invalid_manifest, []}} = Manifest.validate(header, [])
    assert {:error, {:invalid_exact, :yes}} = Manifest.validate(header, %{}, exact: :yes)
  end

  defp fixture_file do
    path = tmp_path("manifest.safetensors")

    Writer.write!(
      %{
        "encoder.weight" => %{dtype: :bf16, shape: [2, 2], data: <<0::size(64)>>},
        "head.bias" => %{dtype: :f32, shape: [2], data: <<0::size(64)>>}
      },
      path,
      metadata: %{"format" => "pt"}
    )

    path
  end

  defp tmp_path(name) do
    dir = Path.join(System.tmp_dir!(), "crucible_safetensors_manifest_tests")
    File.mkdir_p!(dir)
    Path.join(dir, "#{System.unique_integer([:positive])}_#{name}")
  end
end
