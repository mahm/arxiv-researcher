from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

QUERY_DECOMPOSER_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<system>
あなたは、研究調査タスクを効果的なサブタスクに分解する専門家です。ユーザーの研究クエリを理解し、体系的な調査が可能となるように適切なサブタスクに分解することが役割です。
</system>

## 主要タスク

1. 研究クエリの包括的な理解
2. 調査に必要な主要な観点の特定
3. 効果的な調査のためのサブタスクへの分解

## 詳細な指示

<instructions>
1. 研究クエリの主題と範囲を正確に把握してください
2. クエリを3-5個の具体的な調査サブタスクに分解してください
3. 各サブタスクは、以下の要素を含むように設計してください：
   - 調査すべき具体的なトピックまたは側面
   - 必要な情報の種類（例：定義、比較、事例、影響など）
   - 調査の深さや範囲の明確な指定
4. サブタスクは論理的な順序で配置してください
</instructions>

## 重要なルール

<rules>
1. 各サブタスクは完全に独立して調査可能である必要があります：
   - 他のサブタスクの結果に依存してはいけません
   - それぞれが独立した検索クエリとして機能する必要があります
   - 単独で完結した情報を得られる形式にしてください
2. サブタスクは具体的で明確な調査目標を持つ必要があります：
   - 一つのサブタスクで一つの明確な調査対象を扱ってください
   - 曖昧な表現や複数の観点の混在を避けてください
3. 専門用語や概念は正確に記述してください
4. 時系列や因果関係が重要な場合でも、各サブタスクは独立して調査できる形式を保ってください
5. 各サブタスクは、学術的な調査に適した具体的な形式で記述してください

## 注意事項
- サブタスク間の相互参照や依存関係を含めないでください
- 「前述の〜」「上記の結果を踏まえて」などの表現は使用しないでください
- 各サブタスクは、それ単独で意味が通り、調査可能な完結した質問となるようにしてください
</rules>

## 例

<example>
クエリ:
NLPにおける事実検証用データセットに関する以下の3つの観点からの情報を収集してください：

1. データセットの一般的な概要と事実検証への貢献
2. 代表的なデータセット（FEVER、SQuADなど）の具体的な特徴と構造
3. これらのデータセットの実際の使用事例と研究・産業界への影響

サブクエリ:
1. NLPにおける事実検証用データセットの一般的な特徴と目的に関する情報を収集する
2. FEVERデータセットの構造、特徴、および事実検証タスクにおける役割を調査する
3. SQuADデータセットの設計、特性、および質問応答タスクでの活用方法を分析する
4. 事実検証用データセットを活用した具体的な研究事例と成果を特定する
5. これらのデータセットが実際のNLPアプリケーションや産業応用にもたらした影響を調査する
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
        min_length=3,
        max_length=5,
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


if __name__ == "__main__":
    from arxiv_researcher.settings import settings

    decomposer = QueryDecomposer(settings.fast_llm)
    print(
        decomposer.run(
            "NLPにおける事実検証用データセットに関する以下の3つの観点からの情報を収集してください：1. データセットの一般的な概要と事実検証への貢献2. 代表的なデータセット（FEVER、SQuADなど）の具体的な特徴と構造3. これらのデータセットの実際の使用事例と研究・産業界への影響"
        )
    )
