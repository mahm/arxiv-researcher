from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def search_task(self, goal_setting: str, query: str) -> list[str]:
        return self.searcher.run(goal_setting=goal_setting, query=query)

    def run(
        self, goal_setting: str, tasks: list[str], max_workers: int = 5
    ) -> list[str]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.search_task, goal_setting, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                try:
                    task = future_to_task[future]
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    print(f"タスク '{task}' の実行中にエラーが発生しました: {e}")
        return results


if __name__ == "__main__":
    from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
    from arxiv_researcher.settings import settings
    from arxiv_researcher.ui.types import Message

    def on_search_progress(message: Message):
        print(f"検索進捗: {message.content.content}")

    event_emitter = EventEmitter()
    event_emitter.on("search_progress", on_search_progress)

    searcher = ArxivSearcher(settings.llm, event_emitter, max_results=10)

    executor = TaskExecutor(searcher, settings.llm)
    results = executor.run(goal_setting="", tasks=["量子コンピューティング"])
    print(results)
