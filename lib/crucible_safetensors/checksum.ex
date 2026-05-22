defmodule CrucibleSafetensors.Checksum do
  @moduledoc "Checksum helpers for artifact and fixture verification."

  @doc "Returns the lowercase SHA-256 hex digest for a file."
  @spec file_sha256(Path.t()) :: {:ok, String.t()} | {:error, File.posix()}
  def file_sha256(path) when is_binary(path) do
    with {:ok, bytes} <- File.read(path) do
      {:ok, :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower)}
    end
  end

  @doc "Returns the lowercase SHA-256 hex digest for a file, raising on read errors."
  @spec file_sha256!(Path.t()) :: String.t()
  def file_sha256!(path) when is_binary(path) do
    path
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end
end
