from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI

from arxiv_researcher.agent.arxiv_searcher import ArxivPaper, ArxivSearcher


class TaskExecutor:
    RELEVANCE_SCORE_THRESHOLD = 0.7

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.searcher = ArxivSearcher(llm=llm, debug=True)

    def search_task(self, goal_setting: str, query: str) -> list[ArxivPaper]:
        return self.searcher.run(goal_setting=goal_setting, query=query)

    def run(
        self, goal_setting: str, tasks: list[str], max_workers: int = 5
    ) -> list[ArxivPaper]:
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
        # 関連度がしきい値以上の結果のみを返す
        return [
            paper
            for paper in results
            if paper.relevance_score >= self.RELEVANCE_SCORE_THRESHOLD
        ]


if __name__ == "__main__":
    from arxiv_researcher.settings import settings

    executor = TaskExecutor(settings.llm)
    results = executor.run(
        goal_setting="", tasks=["量子コンピューティング", "深層学習"]
    )
    print(results)
