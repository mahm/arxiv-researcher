import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI

from arxiv_researcher.agent.event_emitter import EventEmitter
from arxiv_researcher.searcher.searcher import Searcher
from arxiv_researcher.settings import settings


class TaskExecutor:
    def __init__(
        self,
        llm: ChatOpenAI,
        searcher: Searcher,
    ):
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
    ) -> list[str]:
        results = []
        semaphore = asyncio.Semaphore(max_workers)

        async def search_task(user_hearing: str, query: str) -> list[str]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self.searcher.run,
                user_hearing,
                query,
            )

        async def bounded_search(task):
            async with semaphore:
                try:
                    result = await search_task(user_hearing, task)
                    results.extend(result)
                except Exception as e:
                    print(f"タスク '{task}' の実行中にエラーが発生しました: {e}")

        await asyncio.gather(*[bounded_search(task) for task in tasks])
        return results


if __name__ == "__main__":
    from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
    from arxiv_researcher.ui.types import Message

    def on_search_progress(message: Message):
        print(f"検索進捗: {message.content.content}")

    event_emitter = EventEmitter()
    event_emitter.on("search_progress", on_search_progress)

    searcher = ArxivSearcher(settings.llm, event_emitter, max_results=10)

    executor = TaskExecutor(settings.llm, searcher)
    results = executor.run(user_hearing="", tasks=["量子コンピューティング"])
    print(results)
