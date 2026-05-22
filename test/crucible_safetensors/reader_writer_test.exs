defmodule CrucibleSafetensors.ReaderWriterTest do
  use ExUnit.Case, async: true

  alias CrucibleSafetensors.{Checksum, ChunkReader, Errors, Reader, Writer}

  test "writes and reads a valid multi-tensor file" do
    path = tmp_path("valid.safetensors")

    assert {:ok, ^path} =
             Writer.write(
               %{
                 "b" => %{dtype: :i32, shape: [2], data: <<1::little-32, 2::little-32>>},
                 "a" => %{dtype: :f32, shape: [1], data: <<0, 0, 128, 63>>}
               },
               path,
               metadata: %{"source" => "test"}
             )

    assert {:ok, header} = Reader.open(path)
    assert header.metadata == %{"source" => "test"}
    assert {:ok, tensor} = Reader.tensor(header, "b")
    assert tensor.shape == [2]
    assert tensor.dtype == :i32
    assert {:ok, slice} = Reader.read_tensor(header, tensor)
    assert slice.data == <<1::little-32, 2::little-32>>
    assert {:ok, sha} = Checksum.file_sha256(path)
    assert byte_size(Base.decode16!(sha, case: :lower)) == 32
  end

  test "rejects invalid header length" do
    path = tmp_path("bad_header_len.safetensors")
    File.write!(path, <<999::unsigned-little-64, "{}">>)

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "header length"
  end

  test "rejects invalid header JSON" do
    path = tmp_path("bad_json.safetensors")
    File.write!(path, [<<1::unsigned-little-64>>, "x"])

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "invalid SafeTensors header JSON"
  end

  test "rejects overlapping tensor offsets" do
    header = %{
      "a" => %{"dtype" => "I32", "shape" => [2], "data_offsets" => [0, 8]},
      "b" => %{"dtype" => "I32", "shape" => [2], "data_offsets" => [4, 12]}
    }

    path = raw_file("overlap.safetensors", header, <<0::size(96)>>)

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "overlap"
  end

  test "rejects tensor data shorter than header declares" do
    header = %{
      "a" => %{"dtype" => "I32", "shape" => [2], "data_offsets" => [0, 8]}
    }

    path = raw_file("short.safetensors", header, <<0::little-32>>)

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "invalid data_offsets"
  end

  test "rejects dtype and shape mismatch" do
    header = %{
      "a" => %{"dtype" => "I64", "shape" => [2], "data_offsets" => [0, 8]}
    }

    path = raw_file("mismatch.safetensors", header, <<0::size(64)>>)

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "shape/dtype require"
  end

  test "writer output is deterministic" do
    tensors = %{
      "z" => %{dtype: :i32, shape: [1], data: <<9::little-32>>},
      "a" => %{dtype: :i32, shape: [1], data: <<1::little-32>>}
    }

    one = tmp_path("one.safetensors")
    two = tmp_path("two.safetensors")

    Writer.write!(tensors, one)
    Writer.write!(Map.new(Enum.reverse(Map.to_list(tensors))), two)

    assert File.read!(one) == File.read!(two)
  end

  test "chunked row slices match whole tensor bytes" do
    path = tmp_path("rows.safetensors")
    data = for value <- 1..8, into: <<>>, do: <<value::little-32>>
    Writer.write!(%{"rows" => %{dtype: :i32, shape: [4, 2], data: data}}, path)
    header = Reader.open!(path)
    {:ok, tensor} = Reader.tensor(header, "rows")
    {:ok, whole} = Reader.read_tensor(header, tensor)

    chunked =
      header
      |> ChunkReader.row_slices(tensor, 2)
      |> Enum.map_join(& &1.data)

    assert chunked == whole.data
  end

  defp raw_file(name, header, payload) do
    path = tmp_path(name)
    json = Jason.encode!(header)
    File.write!(path, [<<byte_size(json)::unsigned-little-64>>, json, payload])
    path
  end

  defp tmp_path(name) do
    dir = Path.join(System.tmp_dir!(), "crucible_safetensors_tests")
    File.mkdir_p!(dir)
    Path.join(dir, "#{System.unique_integer([:positive])}_#{name}")
  end
end
