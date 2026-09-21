defmodule CrucibleSafetensors.ChecksumTest do
  use ExUnit.Case, async: true

  alias CrucibleSafetensors.{Checksum, Errors}

  test "incremental SHA-256 matches crypto hash across small chunks" do
    path = tmp_path("streaming.bin")
    data = :binary.copy(<<0, 1, 2, 3, 4, 5, 6, 7>>, 131_073)
    File.write!(path, data)

    expected = :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

    assert {:ok, ^expected} = Checksum.file_sha256(path, chunk_size: 257)
    assert expected == Checksum.file_sha256!(path, chunk_size: 509)
  end

  test "verifies plain and prefixed SHA-256 values" do
    path = tmp_path("verify.bin")
    File.write!(path, "artifact")
    digest = Checksum.file_sha256!(path)

    assert {:ok, ^digest} = Checksum.verify_file(path, String.upcase(digest))
    assert {:ok, ^digest} = Checksum.verify_file(path, "sha256:" <> digest)

    assert {:error, {:checksum_mismatch, _expected, ^digest}} =
             Checksum.verify_file(path, String.duplicate("0", 64))
  end

  test "rejects invalid digests and invalid chunk sizes" do
    path = tmp_path("invalid.bin")
    File.write!(path, "artifact")

    assert {:error, {:invalid_sha256, "nope"}} = Checksum.verify_file(path, "nope")
    assert {:error, %ArgumentError{}} = Checksum.file_sha256(path, chunk_size: 0)

    assert_raise Errors, ~r/SHA-256 mismatch/, fn ->
      Checksum.verify_file!(path, String.duplicate("0", 64))
    end
  end

  defp tmp_path(name) do
    dir = Path.join(System.tmp_dir!(), "crucible_safetensors_checksum_tests")
    File.mkdir_p!(dir)
    Path.join(dir, "#{System.unique_integer([:positive])}_#{name}")
  end
end
