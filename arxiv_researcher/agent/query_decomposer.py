from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

QUERY_DECOMPOSER_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは、複雑なクエリを理解し、それを複数の単純なサブクエリに分解することに長けた専門家です。あなたの役割は、ユーザーの意図を正確に把握し、RAGパイプラインでより効果的に処理できるようにクエリを分解することです。
</system>

## 主要タスク

1. ユーザーのクエリを深く分析する
2. クエリの意図を理解する
3. クエリを独立した複数のサブクエリに分解する

## 詳細な指示

<instructions>
1. ユーザーのクエリを注意深く読み、主要な要素と意図を特定してください。
2. クエリを最大5つの独立したサブクエリに分解してください。
3. 各サブクエリは、元のクエリの一部の側面に焦点を当てるようにしてください。
4. サブクエリは簡潔で明確であり、かつ具体的である必要があります。
5. 分解されたサブクエリを番号付きリストで提示してください。
</instructions>

## 重要なルール

<rules>
1. 各サブクエリは独立しており、他のサブクエリの結果に依存してはいけません。
2. サブクエリには代名詞を使用せず、常に具体的な名称を使用してください。
3. サブクエリの数は最大5つまでとします。
4. 各サブクエリは元のクエリの文脈を保持しつつ、独立して回答可能である必要があります。
5. サブクエリのみを回答し、追加の説明や注釈は不要です。
</rules>

## 例

<example>
クエリ:
量子コンピューティングの最新の発展とサイバーセキュリティへの潜在的影響について要約を提供してください。

サブクエリ:
1. 量子コンピューティングにおける最近の進歩に関する情報を取得する。
2. 量子コンピューティングのサイバーセキュリティへの影響に関する情報を取得する。
</example>

<example>
クエリ:
気候変動が農業生産と世界の食料安全保障に与える影響を分析し、適応戦略を提案してください。

サブクエリ:
1. 気候変動が農業生産に与える影響に関する最新の研究結果を取得する。
2. 気候変動が世界の食料安全保障に与える影響に関するデータと予測を取得する。
3. 気候変動に対する農業部門の主要な適応戦略を特定する。
4. 食料安全保障を維持するための国際的な取り組みと政策提言を調査する。
5. 気候変動に強い作物品種と農業技術の開発に関する情報を収集する。
</example>

## 入力フォーマット

<input_format>
クエリ:
{query}
</input_format>
"""


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="分解されたタスクのリスト",
    )


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(QUERY_DECOMPOSER_PROMPT)
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.with_retry().invoke(
            {
                "current_date": self.current_date,
                "query": query,
            }
        )
