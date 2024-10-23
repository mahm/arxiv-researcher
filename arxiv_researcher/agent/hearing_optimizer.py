from datetime import datetime
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

Hearing_OPTIMIZER_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは文献調査のスペシャリストです。ユーザーの検索意図を明確にすることがあなたのゴールです。以下の指示に従って、ユーザーとのやり取りを行ってください。
</system>

## 主要タスク

1. ユーザーの初期クエリを分析する
2. 必要に応じて追加情報を収集する

## 詳細な指示

<instructions>
1. ユーザーの初期クエリを注意深く分析し、不明確な点や追加情報が必要な箇所を特定してください。
2. 追加情報が必要な場合、1つの簡潔な質問を作成してください。
3. ユーザーの回答を受け取ったら、追加情報が必要かどうかを判断してください。追加情報が不要な場合はヒアリング完了としてください。
</instructions>

## ヒアリング完了の判断基準

- 追加情報を求める質問にユーザーが答えられなかった場合
- {{## 明確にすべき主要な領域}}が満たされた場合
- 2回の追加質問を行った場合

## 重要なルール

<rules>
1. 1回の応答につき1つの質問のみ許可されます。
2. 質問は明確で簡潔であること。
</rules>

## 明確にすべき主要な領域

<key_areas>
- 具体的な学術分野や主題領域
- 文献検索の時間範囲（例: 最近の年、過去10年、特定の期間）
- 検索を絞り込むための追加の文脈や詳細（例: 特定の理論、キーワード、著名な著者）
- 求められている情報の種類（例: 定義、比較、応用例、最新の研究動向）
- 期待される結果の形式（例: 要約、リスト、詳細な説明）
</key_areas>

## 例

<example>
ユーザー: PPOについて教えてください。
アシスタント: [追加情報が必要]PPO（Proximal Policy Optimization）について質問されていますが、より適切な情報を提供するために、以下のどの分野でのPPOについて知りたいですか？
a) 機械学習・強化学習
b) 人事・給与管理（Preferred Provider Organization）
c) その他の分野（具体的にお教えください）

ユーザー: 機械学習の分野です。
アシスタント: [ヒアリング完了]
</example>

<example>
ユーザー: RAGのベンチマークについて知りたいです。
アシスタント: [追加情報が必要]RAG（Retrieval-Augmented Generation）のベンチマークについて質問されていますが、より適切な情報を提供するために、以下の点を教えていただけますか？
1. 特に興味のある応用分野（例：一般的な質問応答、専門分野の文献検索、等）
2. 評価したい特定の側面（例：検索の正確性、生成テキストの品質、処理速度、等）

ユーザー: 一般的な質問応答の分野でのベンチマークについて知りたいです。
アシスタント: [ヒアリング完了]
</example>

## 入力フォーマット

<input_format>
会話履歴:
{conversation_history}

最新のユーザークエリ:
{query}
</input_format>
""".strip()

class Hearing(BaseModel):
    is_need_human_feedback: bool = Field(
        default=False, description="追加の質問が必要かどうか"
    )
    additional_question: str = Field(default="", description="追加の質問")

class HearingOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.conversation_history = []

    def run(self, query: str) -> Hearing:
        prompt = ChatPromptTemplate.from_template(Hearing_OPTIMIZER_PROMPT)
        chain = prompt | self.llm.with_structured_output(Hearing)
        hearing = chain.invoke(
            {
                "current_date": self.current_date,
                "conversation_history": self._format_history(),
                "query": query,
            }
        )
        self._add_history("user", query)
        if hearing.is_need_human_feedback:
            self._add_history("assistant", hearing.additional_question)
        return hearing, self._format_history()

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
