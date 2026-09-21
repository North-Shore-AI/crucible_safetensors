defmodule CrucibleSafetensors.Checksum do
  @moduledoc "Streaming checksum helpers for artifact and fixture verification."

  alias CrucibleSafetensors.Errors

  @default_chunk_size 1_048_576
  @sha256_bytes 32

  @doc "Returns the lowercase SHA-256 hex digest for a file using incremental reads."
  @spec file_sha256(Path.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def file_sha256(path, opts \\ []) when is_binary(path) and is_list(opts) do
    {:ok, file_sha256!(path, opts)}
  rescue
    exception in [File.Error, ArgumentError] -> {:error, exception}
  end

  @doc "Returns the lowercase SHA-256 hex digest for a file, raising on read errors."
  @spec file_sha256!(Path.t(), keyword()) :: String.t()
  def file_sha256!(path, opts \\ []) when is_binary(path) and is_list(opts) do
    chunk_size = chunk_size!(opts)

    File.open!(path, [:read, :binary, :raw], fn file ->
      file
      |> hash_chunks(chunk_size, :crypto.hash_init(:sha256), path)
      |> Base.encode16(case: :lower)
    end)
  end

  @doc "Verifies a file against a SHA-256 digest and returns the actual digest on success."
  @spec verify_file(Path.t(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def verify_file(path, expected, opts \\ []) when is_binary(path) and is_list(opts) do
    with {:ok, expected} <- normalize_sha256(expected),
         {:ok, actual} <- file_sha256(path, opts) do
      if actual == expected do
        {:ok, actual}
      else
        {:error, {:checksum_mismatch, expected, actual}}
      end
    end
  end

  @doc "Verifies a SHA-256 digest, raising when the expected digest is invalid or mismatched."
  @spec verify_file!(Path.t(), String.t(), keyword()) :: String.t()
  def verify_file!(path, expected, opts \\ []) do
    case verify_file(path, expected, opts) do
      {:ok, actual} ->
        actual

      {:error, {:checksum_mismatch, expected, actual}} ->
        raise Errors, "SHA-256 mismatch for #{path}: expected #{expected}, got #{actual}"

      {:error, {:invalid_sha256, value}} ->
        raise ArgumentError, "invalid SHA-256 digest: #{inspect(value)}"

      {:error, %File.Error{} = error} ->
        raise error

      {:error, error} ->
        raise Errors, "SHA-256 verification failed for #{path}: #{inspect(error)}"
    end
  end

  defp hash_chunks(file, chunk_size, context, path) do
    case :file.read(file, chunk_size) do
      {:ok, bytes} ->
        hash_chunks(file, chunk_size, :crypto.hash_update(context, bytes), path)

      :eof ->
        :crypto.hash_final(context)

      {:error, reason} ->
        raise File.Error, reason: reason, action: "read", path: path
    end
  end

  defp chunk_size!(opts) do
    case Keyword.get(opts, :chunk_size, @default_chunk_size) do
      value when is_integer(value) and value > 0 ->
        value

      value ->
        raise ArgumentError, "chunk_size must be a positive integer, got: #{inspect(value)}"
    end
  end

  defp normalize_sha256(value) when is_binary(value) do
    digest = value |> String.trim() |> strip_sha256_prefix() |> String.downcase()

    with true <- byte_size(digest) == @sha256_bytes * 2,
         {:ok, _bytes} <- Base.decode16(digest, case: :mixed) do
      {:ok, digest}
    else
      _ -> {:error, {:invalid_sha256, value}}
    end
  end

  defp normalize_sha256(value), do: {:error, {:invalid_sha256, value}}

  defp strip_sha256_prefix("sha256:" <> digest), do: digest
  defp strip_sha256_prefix(digest), do: digest
end
