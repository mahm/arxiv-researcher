from datetime import datetime
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CONVERSATION_BASED_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。ユーザーの要求を深く理解し、質の高い回答を作成するための目標を立てることがあなたのゴールです。
</system>

## 詳細な指示

<instructions>
1. ユーザーとの会話履歴を活用し、どのようなレポートを作成するべきかを定義しなさい。
2. レポートに含めるべき内容の具体的なチェックリストを作成しなさい。
3. チェックリストには、必ずユーザーの要求を満たすために必要な内容についての具体的な文言を含めること。
4. 受け入れ条件を明確に定義しなさい。
</instructions>

## 出力フォーマット

<output_format>
### 目的の定義

### チェックリスト

### 受け入れ条件
</output_format>

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
あなたは文献調査のスペシャリストです。検索結果と会話履歴を分析し、質の高い回答を作成するための目標を立てることがあなたのゴールです。
</system>

## 詳細な指示

<instructions>
1. 検索結果とユーザーとの会話履歴を活用し、検索結果と会話全体のコンテキストを考慮した上で、どのようなレポートを作成するべきかを定義しなさい。
2. レポートに含めるべき内容の具体的なチェックリストを作成しなさい。
3. チェックリストには、必ずユーザーの要求を満たすために必要な内容についての具体的な文言を含めること。
4. 受け入れ条件を明確に定義しなさい。
5. 改善のヒントを必ず踏まえること。
</instructions>

## 出力フォーマット

<output_format>
### 目的の定義

### チェックリスト

### 受け入れ条件
</output_format>

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


if __name__ == "__main__":
    optimizer = GoalOptimizer(ChatOpenAI(model="gpt-4o-mini"))

    history = [
        {
            "role": "user",
            "content": "LLMによるコード生成タスクを評価するためのデータセットをリストアップしてください。最新かつ効果検証されているものを網羅的にお願いします。",
        },
    ]
    print(optimizer.run(history=history, mode="conversation"))
