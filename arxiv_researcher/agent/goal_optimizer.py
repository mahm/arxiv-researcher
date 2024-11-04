from datetime import datetime
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

GOAL_OPTIMIZER_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。ユーザーの質問を深く理解し、質の高い回答を作成するための計画を作成することがあなたのゴールです。以下の指示に従って、ユーザーとのやり取りを行ってください。
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

## 例

<example>
会話履歴:
ユーザー: 事実に関する質問に対するモデルの回答の正確性を検証するためのデータセットを推薦してください。
アシスタント: [追加情報が必要]どの学術分野や領域のデータセットにご興味がありますか？

ユーザー: コンピュータサイエンスのNLP
アシスタント: [作成するレポート内容を定義]

### データセットの概要

NLPにおける事実検証に特化したデータセットの概要を説明します。データセットの種類、主な用途、およびそのデータセットがどのようにして事実検証に貢献するかに焦点を当てます。一覧で分かりやすいように表形式でまとめます。他の段落で詳細に説明する具体的なデータセットの名前や特性には触れません。

### データセットの例と詳細

具体的なデータセット（例：FEVER, SQuADなど）を紹介し、それぞれのデータセットの詳細な特徴や提供する情報の種類について説明します。ここでは、それぞれのデータセットがどのように構築され、どのような問題に対応しているかを明らかにします。一覧で分かりやすいように表形式でまとめます。段落1で紹介した一般的な情報を繰り返さないようにします。

### データセットの使用事例と影響

選ばれたデータセットが実際にどのように使用されているか、およびこれが研究や業界にどのような影響を与えているかを詳述します。具体的な研究事例やプロジェクトを挙げ、データセットの実用性とその成果に焦点を当てます。一覧で分かりやすいように表形式でまとめます。段落2で触れた具体的なデータの特性に基づく使用事例を提供し、その効果を強調します。
</example>

## 入力フォーマット

<input_format>
会話履歴:
{conversation_history}
</input_format>
""".strip()


class Goal(BaseModel):
    content: str = Field(default="", description="計画の内容")


class GoalOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, history: list) -> Goal:
        prompt = ChatPromptTemplate.from_template(GOAL_OPTIMIZER_PROMPT)
        chain = prompt | self.llm.with_structured_output(Goal)
        goal = chain.invoke(
            {
                "current_date": self.current_date,
                "conversation_history": self._format_history(history),
            }
        )
        return goal

    def _format_history(self, history):
        return "\n".join(
            [f"{message['role']}: {message['content']}" for message in history]
        )
