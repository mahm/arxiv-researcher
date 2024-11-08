import datetime
import os
import urllib.parse
from typing import Optional

import cohere
import feedparser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from arxiv_researcher.agent.event_emitter import EventEmitter
from arxiv_researcher.searcher.searcher import Searcher
from arxiv_researcher.settings import settings
from arxiv_researcher.ui.types import Message, SearchProgress

FIELD_SELECTOR_PROMPT = """\
Determine the arXiv categories that need to be searched based on the user's query.
Select one or more category names, separated by commas.
Reply only with the exact category names (e.g., cs.AI, math.CO).

User Query: {query}
""".strip()

DATE_SELECTOR_PROMPT = """\
Determine the time range to be retrieved based on the user's query and the current system time.
Use the format YYMM-YYMM (e.g., 2203-2402 for March 2022 to February 2024).
If no time range is specified, reply with "NONE".

Current Date: {current_date}
User Query: {query}
""".strip()

EXPAND_QUERY_PROMPT = """\
<system>
あなたは、与えられた単一のサブクエリから効果的なarXiv検索クエリを生成する専門家です。あなたの役割は、学術的な文脈を理解し、arXivの検索システムで直接使用できる最適な検索クエリを作成することです。

{feedback}
</system>

## 主要タスク

1. 提供されたサブクエリを分析する
2. サブクエリから重要なキーワードを抽出する
3. 抽出したキーワードを使用して、arXivで直接使用可能な効果的な検索クエリを構築する

## 詳細な指示

<instructions>
1. サブクエリを注意深く読み、主要な概念や専門用語を特定してください。
2. 学術的文脈に適した具体的なキーワードを選択してください。
3. 同義語や関連する用語も考慮に入れてください。
4. arXivの検索構文を適切に使用して、効果的な検索クエリを作成してください。
5. 検索結果が適切に絞り込まれるよう、必要に応じてフィールド指定子を使用してください。
6. 生成したクエリがarXivの検索ボックスに直接コピー＆ペーストできることを確認してください。
</instructions>

## 重要なルール

<rules>
1. クエリには1〜2つの主要なキーワードまたはフレーズを含めてください。
2. 一般的すぎる用語や非学術的な用語は避けてください。
3. 検索クエリは80文字以内に収めてください。
4. クエリの前後に余分な空白や引用符を付けないでください。
5. 説明や理由付けは含めず、純粋な検索クエリのみを出力してください。
6. シンプルなクエリを心がけてください。
</rules>

## arXiv検索の構文ヒント

<arxiv_syntax>
- AND: 複数の用語を含む文書を検索（例：quantum AND computing）
- OR: いずれかの用語を含む文書を検索（例：neural OR quantum）
- 引用符: フレーズ検索（例："quantum computing"）
- フィールド指定子: ti:（タイトル）, au:（著者）, abs:（要約）
- マイナス記号: 特定の用語を除外（例：quantum -classical）
- ワイルドカード: 部分一致検索（例：neuro*）
</arxiv_syntax>

## 例

<example>
クエリ: 量子コンピューティングにおける最近の進歩に関する情報を取得する。

arXiv検索クエリ:
ti:"quantum computing" AND ("recent advances" OR "latest developments") -review
</example>

<example>
クエリ: 深層強化学習の金融市場への応用に関する最新の研究を見つける。

arXiv検索クエリ:
"deep reinforcement learning" AND "financial markets" AND (application OR implementation)
</example>

## 入力フォーマット

<input_format>
目標: {goal_setting}
クエリ: {query}
</input_format>

## 出力フォーマット

<output_format>
[生成されたarXiv検索クエリ]
</output_format>
""".strip()


class ArxivFields(BaseModel):
    values: list[str] = Field(
        description="The arXiv categories that need to be searched based on the user's query."
    )


class ArxivTimeRange(BaseModel):
    start: Optional[datetime.datetime] = Field(
        default=None,
        description="The start date of the time range to be retrieved.",
    )
    end: Optional[datetime.datetime] = Field(
        default=None, description="The end date of the time range to be retrieved."
    )

    @property
    def text(self) -> Optional[str]:
        if self.start and self.end:
            return f"{self.start.strftime('%Y%m%d')}+TO+{self.end.strftime('%Y%m%d')}"
        elif self.start:
            return f"{self.start.strftime('%Y%m%d')}+TO+LATEST"
        elif self.end:
            return f"EARLIEST+TO+{self.end.strftime('%Y%m%d')}"
        return None


class ArxivPaper(BaseModel):
    id: str
    title: str
    link: str
    pdf_link: str
    summary: str
    published: datetime.datetime
    updated: datetime.datetime
    version: int
    authors: list[str]
    categories: list[str]
    relevance_score: Optional[float] = None

    @property
    def text(self) -> str:
        return f"""\
<paper>
  <id>{self.id}</id>
  <title>{self.title}</title>
  <link>{self.link}</link>
  <pdf_link>{self.pdf_link}</pdf_link>
  <summary>{self.summary}</summary>
  <published>{self.published}</published>
  <updated>{self.updated}</updated>
  <version>{self.version}</version>
  <authors>{', '.join(self.authors)}</authors>
  <categories>{', '.join(self.categories)}</categories>
  {f"<relevance_score>{self.relevance_score}</relevance_score>" if self.relevance_score else ""}
</paper>"""


class ArxivSearcher(Searcher):
    RELEVANCE_SCORE_THRESHOLD = 0.7

    def __init__(
        self,
        llm: ChatOpenAI,
        event_emitter: EventEmitter,
        cohere_client: cohere.Client = settings.cohere_client,
        max_results: int = settings.max_search_results,
        debug: bool = settings.debug,
    ):
        self.llm = llm
        self.event_emitter = event_emitter
        self.cohere_client = cohere_client
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.max_results = max_results
        self.debug = debug

    def _field_selector(self, query: str) -> ArxivFields:
        prompt = ChatPromptTemplate.from_template(FIELD_SELECTOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(ArxivFields)
        return chain.invoke({"query": query})

    def _date_selector(self, query: str) -> ArxivTimeRange:
        prompt = ChatPromptTemplate.from_template(DATE_SELECTOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(ArxivTimeRange)
        return chain.invoke(
            {
                "current_date": self.current_date,
                "query": query,
            }
        )

    def _expand_query(self, goal_setting: str, query: str, feedback: str = "") -> str:
        prompt = ChatPromptTemplate.from_template(EXPAND_QUERY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"goal_setting": goal_setting, "query": query, "feedback": feedback}
        )

    def run(self, goal_setting: str, query: str) -> list[dict]:
        base_url = "https://export.arxiv.org/api/query?search_query="
        max_retries = 3
        retry_count = 0
        feedback = ""
        papers = []

        while retry_count < max_retries:
            arxiv_fields: ArxivFields = self._field_selector(query)
            filterquery_str = (
                " AND ".join([f"cat:{item}" for item in arxiv_fields.values])
                if arxiv_fields.values
                else ""
            )

            arxiv_time_range: ArxivTimeRange = self._date_selector(query)
            query_filterdate = arxiv_time_range.text

            expanded_query = self._expand_query(goal_setting, query, feedback)

            search_query = (
                f"{filterquery_str} AND all:{expanded_query}"
                if filterquery_str
                else f"all:{expanded_query}"
            )
            encoded_search_query = urllib.parse.quote(search_query)

            full_url = f"{base_url}{encoded_search_query}&sortBy=relevance&max_results={self.max_results}"
            if query_filterdate:
                full_url += f"&submittedDate={query_filterdate}"
            if self.debug:
                print(f"Searching for papers: {full_url}")

            feed = feedparser.parse(full_url)
            entries = feed.entries

            papers = [
                ArxivPaper(
                    id=entry.id.split("/")[-1].split("v")[0],
                    title=entry.title,
                    link=entry.link,
                    pdf_link=next(
                        (
                            link.href
                            for link in entry.links
                            if link.type == "application/pdf"
                        ),
                        None,
                    ),
                    summary=entry.summary.replace("\n", " "),
                    published=datetime.datetime(*entry.published_parsed[:6]),
                    updated=datetime.datetime(*entry.updated_parsed[:6]),
                    version=int(entry.id.split("/")[-1].split("v")[-1]),
                    authors=[
                        author.get("name", "") for author in entry.get("authors", [])
                    ],
                    categories=[tag.get("term", "") for tag in entry.get("tags", [])],
                )
                for entry in entries
            ]

            if self.debug:
                print(f"Found {len(papers)} papers.")

            if papers:
                if self.debug:
                    print("Papers found. Exiting retry loop.")
                self._notify(
                    SearchProgress(
                        role="assistant",
                        task=f"({len(papers)}件){query}",
                        content=(
                            f"検索クエリ: {expanded_query}\n" f"検索URL: {full_url}"
                        ),
                    ),
                ),
                break  # 結果が見つかったのでループを抜ける

            else:
                retry_count += 1
                if retry_count < max_retries:
                    feedback = f"検索結果が0件でした。クエリをより一般的なものや関連するキーワードに調整してください。"
                    if self.debug:
                        print(
                            f"No papers found. Retrying with adjusted query. Attempt {retry_count}/{max_retries}"
                        )
                else:
                    if self.debug:
                        print("Max retries reached. No results found.")
                    self._notify(
                        SearchProgress(
                            role="assistant",
                            task=f"(0件){query}",
                            content=(
                                f"検索クエリ: {expanded_query}\n" f"検索URL: {full_url}"
                            ),
                        ),
                    ),
                    break  # 最大リトライ回数に達したのでループを抜ける

        if papers:
            reranked = self.cohere_client.rerank(
                model=settings.cohere_rerank_model,
                query=f"{goal_setting}\n{query}",
                documents=[f"{paper.title}\n{paper.summary}" for paper in papers],
                top_n=min(10, len(papers)),
            )

            reranked_papers = []
            for result in reranked.results:
                paper = papers[result.index]
                paper.relevance_score = result.relevance_score
                reranked_papers.append(paper)

            # 関連度がしきい値以上の結果のみを返す
            papers = [
                paper.model_dump()
                for paper in reranked_papers
                if paper.relevance_score >= self.RELEVANCE_SCORE_THRESHOLD
            ]

        return papers

    def _notify(self, search_progress: SearchProgress) -> None:
        self.event_emitter.emit(
            "search_progress",
            Message(content=search_progress),
        )


def main():
    from arxiv_researcher.settings import settings

    searcher = ArxivSearcher(settings.llm, settings.cohere_client, debug=True)

    query = input("Enter your arXiv search query: ")
    results = searcher.run(goal_setting="", query=query)

    print(f"\nFound {len(results)} results:")
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. Title: {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Summary: {paper.summary[:500]}...")
        print(f"   arXiv ID: {paper.id}")
        print(f"   PDF Link: {paper.pdf_link}")
        print(f"   Relevance Score: {paper.relevance_score:.4f}")


if __name__ == "__main__":
    main()
