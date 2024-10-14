# arXiv Researcher

arXiv ResearcherはarXivのAPIを利用し、自然言語での質問に基づいて、関連する論文を検索し、その内容を簡潔にまとめます。

## セットアップ

※ [uv](https://github.com/astral-sh/uv)を利用しています。uvをインストールしていない場合は、以下のコマンドでインストールしてください。

```
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

本プロジェクトのセットアップは次のコマンドで行えます。

```
$ uv venv
$ uv sync
```

UIの起動は次のコマンドで行えます。

```
$ make run
```
