import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI

from arxiv_researcher.agent.event_emitter import EventEmitter
from arxiv_researcher.searcher.searcher import Searcher
from arxiv_researcher.service.arxiv_rag import ArxivRAG
from arxiv_researcher.settings import settings


class TaskExecutionError(Exception):
    pass


class SearchError(Exception):
    pass


class RAGError(Exception):
    pass


class TaskExecutor:
    def __init__(
        self,
        llm: ChatOpenAI,
        searcher: Searcher,
    ) -> None:
        self.llm = llm
        self.searcher = searcher
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)

    def run(
        self,
        user_hearing: str,
        tasks: list[str],
        max_workers: int = settings.max_workers,
    ) -> list[str]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.arun(user_hearing, tasks, max_workers))
        finally:
            loop.close()

    async def arun(
        self,
        user_hearing: str,
        tasks: list[str],
        max_workers: int = settings.max_workers,
    ) -> list[dict]:
        results: list[dict] = []
        semaphore = asyncio.Semaphore(max_workers)

        try:
            await asyncio.gather(
                *[
                    self._process_task(task, user_hearing, semaphore, results)
                    for task in tasks
                ]
            )
            return results
        except Exception as e:
            raise TaskExecutionError(f"タスクの実行中にエラーが発生しました: {e}")

    async def _process_task(
        self,
        task: str,
        user_hearing: str,
        semaphore: asyncio.Semaphore,
        results: list[dict],
    ) -> None:
        # LLMに検索が必要かどうかを判断させる
        needs_search = await self._check_if_search_needed(task)

        if needs_search:
            await self._bounded_search(task, user_hearing, semaphore, results)
        else:
            # LLM単体で回答を生成
            direct_answer = await self._get_direct_llm_answer(task)
            results.append(
                {
                    "answer": direct_answer,
                    "title": "Direct Answer from LLM",
                    "pdf_link": None,
                    "summary": None,
                }
            )

    async def _check_if_search_needed(self, task: str) -> bool:
        prompt = f"""\
以下の質問に対して、arXiv論文の検索が必要かどうかを判断してください。
        
質問: {task}

以下の場合は論文検索が必要です：
- 最新の研究動向や具体的な研究事例が必要な場合
- 特定の技術や手法の詳細な実装例が必要な場合
- 実験結果や比較データが必要な場合

以下の場合は論文検索が不要です：
- 一般的な概念の説明で十分な場合
- 基本的な定義や説明で回答可能な場合
- 広く知られている情報で回答可能な場合

回答は "True" または "False" のみを返してください。
        """.strip()

        limited_llm = self.llm.with_config({"max_tokens": 1})
        response = await limited_llm.ainvoke(prompt)
        return response.content.strip().lower() == "true"

    async def _get_direct_llm_answer(self, task: str) -> str:
        prompt = f"""\
以下の質問に対して、あなたの知識に基づいて回答してください：

質問: {task}
        """.strip()

        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def _bounded_search(
        self,
        task: str,
        user_hearing: str,
        semaphore: asyncio.Semaphore,
        results: list[dict],
    ) -> None:
        async with semaphore:
            try:
                result = await self._search_and_analyze_task(user_hearing, task)
                results.extend(result)
            except Exception as e:
                print(f"タスク '{task}' の実行中にエラーが発生しました: {e}")

    async def _search_and_analyze_task(
        self, user_hearing: str, query: str
    ) -> list[dict]:
        search_results = await self._search_papers(user_hearing, query)
        pdf_urls = self._extract_pdf_urls(search_results)
        return await self._analyze_papers(pdf_urls, query, search_results)

    async def _search_papers(self, user_hearing: str, query: str) -> list[dict]:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor, self.searcher.run, user_hearing, query
            )
        except Exception as e:
            raise SearchError(f"論文検索中にエラーが発生しました: {e}")

    def _extract_pdf_urls(self, search_results: list[dict]) -> list[str]:
        return [
            result["pdf_link"] for result in search_results if result.get("pdf_link")
        ]

    async def _analyze_papers(
        self, pdf_urls: list[str], query: str, search_results: list[dict]
    ) -> list[dict]:
        async def process_single_paper(pdf_url: str, search_result: dict) -> dict:
            try:
                answer = await self._run_rag_for_paper(pdf_url, query)
                # [NOT_RELATED]の場合はNoneを返して後でフィルタリング
                if "[NOT_RELATED]" in answer:
                    return None
                return {
                    "pdf_url": pdf_url,
                    "answer": answer,
                    **search_result,
                }
            except Exception as e:
                print(f"RAGの実行中にエラーが発生しました（{pdf_url}）: {e}")
                raise

        tasks = [
            process_single_paper(pdf_url, search_results[i])
            for i, pdf_url in enumerate(pdf_urls)
        ]

        rag_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Noneと例外を除外
        return [
            result
            for result in rag_results
            if result is not None and not isinstance(result, Exception)
        ]

    async def _run_rag_for_paper(self, pdf_url: str, query: str) -> str:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor, lambda: ArxivRAG(pdf_url).run(query)
            )
        except Exception as e:
            raise RAGError(f"RAG実行中にエラーが発生しました: {e}")


if __name__ == "__main__":
    from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
    from arxiv_researcher.ui.types import Message

    def on_search_progress(message: Message):
        print(f"検索進捗: {message.content.content}")

    event_emitter = EventEmitter()
    event_emitter.on("search_progress", on_search_progress)

    searcher = ArxivSearcher(settings.llm, event_emitter, max_results=5)

    executor = TaskExecutor(settings.llm, searcher)
    results = executor.run(
        user_hearing="",
        tasks=["ソフトウェア開発へのAIエージェントの応用について実用例を取得する"],
    )
    print(results)
