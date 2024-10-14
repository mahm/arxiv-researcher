from abc import ABC, abstractmethod


class Searcher(ABC):
    @abstractmethod
    def run(self, goal_setting: str, query: str) -> list[str]:
        pass
