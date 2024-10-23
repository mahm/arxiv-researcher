from datetime import datetime
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

QUERY_REWRITE_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。ユーザーの質問を深く理解し、最適な文献を調査するためのクエリを作成することがあなたのゴールです。以下の指示に従って、ユーザーとのやり取りを行ってください。
</system>

## 主要タスク

1. 会話履歴を考慮してクエリを書き換える

## 詳細な指示

<instructions>
1. 会話全体のコンテキストを考慮してクエリを書き換えてください。
</instructions>

## 重要なルール

<rules>
1. ユーザーの元のクエリの情報を絶対に省略しないこと。
2. 会話で明示的に述べられた情報のみを使用し、推測や仮定を避けること。
3. 書き換えられたクエリは、元のクエリよりも具体的で情報量が多くなるようにすること。
</rules>

<rules>
- ユーザーからの質問内容は一切省略せずに含める必要があります。
- 回答は、ユーザーの視点で書き換えた新しい質問文のみを含めてください。
</rules>

## 例

<example>
ユーザー: 事実に関する質問に対するモデルの回答の正確性を検証するためのデータセットを推薦してください。
アシスタント: [追加情報が必要]どの学術分野や領域のデータセットにご興味がありますか？

ユーザー: コンピュータサイエンスのNLP
アシスタント: [書き換えられたクエリ]NLP（自然言語処理）分野における、事実に関する質問に対するモデルの回答の正確性を検証するためのデータセットを推薦してください。
</example>

## 入力フォーマット

<input_format>
会話履歴:
{conversation_history}
</input_format>
""".strip()

class Goal(BaseModel):
    content: str = Field(default="", description="書き換えられたクエリ")

class GoalOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.conversation_history = []

    def run(self, query: str, history) -> Goal:
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        chain = prompt | self.llm.with_structured_output(Goal)
        query_rewrite = chain.invoke(
            {
                "current_date": self.current_date,
                "conversation_history": history, 
            }
        )
        print("会話履歴", history)
        self._add_history("user", query)
        print("実行の結果", query_rewrite)
        self._add_history("assistant", query_rewrite.content)
        return query_rewrite

    def _add_history(self, role: Literal["user", "assistant"], content: str):
        self.conversation_history.append({"role": role, "content": content})

    def _format_history(self):
        return "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in self.conversation_history
            ]
        )

    def reset(self):
        self.conversation_history = []

