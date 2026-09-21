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
    assert {:error, :not_found} = Reader.tensor(header, "missing")
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
    File.write!(path, [<<1::unsigned-little-64>>, "{"])

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

  test "supports current byte and sub-byte SafeTensors dtype metadata without Nx" do
    cases = [
      {:bool, [3], 3},
      {:i8, [3], 3},
      {:u16, [3], 6},
      {:f8_e4m3, [3], 3},
      {:f4, [4], 2},
      {:f6_e2m3, [4], 3},
      {:f64, [2], 16},
      {:u64, [2], 16}
    ]

    Enum.each(cases, fn {dtype, shape, nbytes} ->
      path = tmp_path("dtype_#{dtype}.safetensors")

      Writer.write!(
        %{"value" => %{dtype: dtype, shape: shape, data: :binary.copy(<<0>>, nbytes)}},
        path
      )

      header = Reader.open!(path)
      assert {:ok, tensor} = Reader.tensor(header, "value")
      assert tensor.dtype == dtype
      assert tensor.nbytes == nbytes
    end)
  end

  test "rejects non-string metadata values" do
    path = tmp_path("invalid_metadata.safetensors")

    assert {:error, %Errors{message: message}} =
             Writer.write(
               %{"a" => %{dtype: :i32, shape: [1], data: <<1::little-32>>}},
               path,
               metadata: %{"count" => 1}
             )

    assert message =~ "metadata must contain only string keys and values"
  end

  test "row slicing rejects sub-byte layouts whose rows are not byte-aligned" do
    path = tmp_path("subbyte_rows.safetensors")
    Writer.write!(%{"rows" => %{dtype: :f4, shape: [2, 3], data: <<0, 0, 0>>}}, path)
    header = Reader.open!(path)
    {:ok, tensor} = Reader.tensor(header, "rows")

    assert {:error, %Errors{message: message}} = Reader.read_row_slice(header, tensor, 0, 1)
    assert message =~ "not byte-aligned"
  end

  test "rejects total sub-byte tensor sizes that are not byte-aligned" do
    path = tmp_path("misaligned_subbyte.safetensors")

    assert {:error, %Errors{message: message}} =
             Writer.write(%{"value" => %{dtype: :f4, shape: [3], data: <<0, 0>>}}, path)

    assert message =~ "not byte-aligned"
  end

  test "rejects duplicate JSON keys in headers" do
    path = tmp_path("duplicate_key.safetensors")

    json =
      ~s({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})

    File.write!(path, [<<byte_size(json)::unsigned-little-64>>, json, <<1::little-32>>])

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "duplicate JSON object key"
  end

  test "rejects payload gaps and trailing payload bytes" do
    gap_header = %{
      "a" => %{"dtype" => "I32", "shape" => [1], "data_offsets" => [4, 8]}
    }

    gap_path = raw_file("gap.safetensors", gap_header, <<0::size(64)>>)
    assert {:error, %Errors{message: gap_message}} = Reader.open(gap_path)
    assert gap_message =~ "starts at 4, expected 0"

    trailing_header = %{
      "a" => %{"dtype" => "I32", "shape" => [1], "data_offsets" => [0, 4]}
    }

    trailing_path = raw_file("trailing.safetensors", trailing_header, <<0::size(64)>>)
    assert {:error, %Errors{message: trailing_message}} = Reader.open(trailing_path)
    assert trailing_message =~ "file contains 8 payload bytes"
  end

  test "rejects headers that do not begin with a JSON object" do
    path = tmp_path("leading_space.safetensors")
    json = ~s( {"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})
    File.write!(path, [<<byte_size(json)::unsigned-little-64>>, json, <<1::little-32>>])

    assert {:error, %Errors{message: message}} = Reader.open(path)
    assert message =~ "must begin with a JSON object"
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
