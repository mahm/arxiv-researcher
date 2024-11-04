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
                    self._bounded_search(task, user_hearing, semaphore, results)
                    for task in tasks
                ]
            )
            return results
        except Exception as e:
            raise TaskExecutionError(f"タスクの実行中にエラーが発生しました: {e}")

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
        return [result for result in rag_results if not isinstance(result, Exception)]

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
