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

### 1. `.env.sample`を`.env`にコピーし、OpenAI API Keyなどを設定してください。

```
$ cp .env.sample .env
```

### 2. 次のコマンドで、必要なパッケージをインストールします。

```
$ uv venv
$ uv sync
```

### 3. 次のコマンドで、UI(Streamlit)を起動します。

```
$ make run
```