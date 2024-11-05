from datetime import datetime
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CONVERSATION_BASED_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。ユーザーの質問を深く理解し、質の高い回答を作成するための計画を立てることがあなたのゴールです。
</system>

## 主要タスク

- 会話履歴を考慮して計画を立てる

## 詳細な指示

<instructions>
会話全体のコンテキストを考慮して、以下のフォーマットで回答を作成するための指示リストを作成してください。

段落1の指示
段落2の指示
...
段落Nの指示
</instructions>

## 重要なルール

<rules>
1. 各段落は独立して並行して書かれるように設計する。
2. 段落の集合が質問に対する回答を形成できるようにする。
3. 各段落がどのような情報を含むべきか、またどの情報を避けるべきかを明確にする。
</rules>

## 入力フォーマット

<input_format>
会話履歴:
{conversation_history}
</input_format>
""".strip()

SEARCH_BASED_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。検索結果と会話履歴を分析し、質の高い回答を作成するための計画を立てることがあなたのゴールです。
</system>

## 主要タスク

- 検索結果と会話履歴を考慮して計画を立てる

## 詳細な指示

<instructions>
検索結果と会話全体のコンテキストを考慮して、以下のフォーマットで回答を作成するための指示リストを作成してください。

段落1の指示
段落2の指示
...
段落Nの指示
</instructions>

## 重要なルール

<rules>
1. 各段落は独立して並行して書かれるように設計する。
2. 段落の集合が質問に対する回答を形成できるようにする。
3. 各段落がどのような情報を含むべきか、またどの情報を避けるべきかを明確にする。
4. 検索結果から得られた具体的な情報を活用する。
</rules>

## 入力フォーマット

<input_format>
会話履歴:
{conversation_history}

検索結果:
{search_results}

改善のヒント:
{improvement_hint}
</input_format>
""".strip()


class GoalOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(
        self,
        history: list,
        mode: Literal["conversation", "search"] = "conversation",
        search_results: list | None = None,
        improvement_hint: str | None = None,
    ) -> str:
        template = (
            SEARCH_BASED_PROMPT if mode == "search" else CONVERSATION_BASED_PROMPT
        )
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        inputs = {
            "current_date": self.current_date,
            "conversation_history": self._format_history(history),
        }

        if mode == "search" and search_results:
            inputs["search_results"] = self._format_search_results(search_results)
        if improvement_hint:
            inputs["improvement_hint"] = improvement_hint

        return chain.invoke(inputs)

    def _format_history(self, history):
        return "\n".join(
            [f"{message['role']}: {message['content']}" for message in history]
        )

    def _format_search_results(self, results: list) -> str:
        return "\n\n".join(
            [
                f"Title: {result.get('title', '')}\n"
                f"Abstract: {result.get('abstract', '')}"
                for result in results
            ]
        )
