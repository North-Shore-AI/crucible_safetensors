defmodule CrucibleSafetensors.Boundary.NoLibOsEnvTest do
  use ExUnit.Case, async: true

  @forbidden ~w(
    System.get_env
    System.fetch_env
    System.fetch_env!
    System.put_env
    System.delete_env
  )

  test "no lib/** source calls direct OS env APIs" do
    assert forbidden_hits() == []
  end

  defp forbidden_hits do
    "lib/**/*.{ex,exs}"
    |> Path.wildcard()
    |> Enum.filter(&(String.ends_with?(&1, ".ex") or String.ends_with?(&1, ".exs")))
    |> Enum.flat_map(&file_hits/1)
  end

  defp file_hits(path) do
    body = File.read!(path)

    Enum.flat_map(@forbidden, fn token ->
      token_hits(path, body, token)
    end)
  end

  defp token_hits(path, body, token) do
    if String.contains?(body, token), do: [{path, token}], else: []
  end
end
