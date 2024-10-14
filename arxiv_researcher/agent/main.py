import operator
from typing import Annotated, Callable, Literal, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.types import StateSnapshot
from pydantic import BaseModel, Field

from arxiv_researcher.agent.event_emitter import EventEmitter
from arxiv_researcher.agent.goal_optimizer import Goal, GoalOptimizer
from arxiv_researcher.agent.query_decomposer import DecomposedTasks, QueryDecomposer
from arxiv_researcher.agent.reporter import Reporter
from arxiv_researcher.agent.task_executor import TaskExecutor
from arxiv_researcher.agent.utility import dict_to_xml_str
from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
from arxiv_researcher.settings import settings
from arxiv_researcher.ui.types import (
    AlertMessage,
    ChatMessage,
    DataframeMessage,
    Message,
)


class ArxivResearcherState(BaseModel):
    human_inputs: Annotated[list[str], operator.add] = Field(default_factory=list)
    goal: Goal = Field(default=None)
    tasks: list[str] = Field(default_factory=list)
    results: list[dict] = Field(default_factory=list)
    final_output: str = Field(default="")


class ArxivResearcher(EventEmitter):
    def __init__(self, llm: ChatOpenAI) -> None:
        super().__init__()
        self.subscribers: list[Callable[[str, Message], None]] = []
        self.goal_optimizer = GoalOptimizer(llm)
        self.query_decomposer = QueryDecomposer(llm)
        self.task_executor = TaskExecutor(
            llm, searcher=ArxivSearcher(llm, event_emitter=self)
        )
        self.reporter = Reporter(llm)
        self.graph = self._create_graph()

    def _notify(
        self,
        event: str,
        message: ChatMessage | DataframeMessage | AlertMessage,
    ) -> None:
        self.emit(event, Message(content=message))

    def subscribe(self, event: str, subscriber: Callable[[str, Message], None]) -> None:
        self.on(event, subscriber)

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        if self.is_next_human_feedback_node(thread_id):
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node="human_feedback",
            )
        else:
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node=START,
            )
        self._stream_events(human_message=human_message, thread_id=thread_id)

    def is_next_human_feedback_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_feedback"

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()

    def reset(self) -> None:
        self.graph.reset()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(ArxivResearcherState)

        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("human_feedback", self._human_feedback)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("terminate", self._terminate)

        graph.add_edge(START, "goal_setting")
        graph.add_conditional_edges("goal_setting", self._route_goal_setting)
        graph.add_edge("human_feedback", "goal_setting")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_conditional_edges("execute_task", self._route_execute_task)
        graph.add_edge("generate_report", "terminate")
        graph.add_edge("terminate", END)

        memory = MemorySaver()

        return graph.compile(
            checkpointer=memory,
            interrupt_before=["human_feedback"],
        )

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _goal_setting(self, state: ArxivResearcherState) -> dict:
        human_input = state.human_inputs[-1]
        goal: Goal = self.goal_optimizer.run(human_input)
        return {"goal": goal}

    def _route_goal_setting(
        self, state: ArxivResearcherState
    ) -> Literal["human_feedback", "decompose_query"]:
        return (
            "human_feedback" if state.goal.is_need_human_feedback else "decompose_query"
        )

    def _human_feedback(self, state: ArxivResearcherState) -> None:
        # ユーザーからの入力を待つ
        pass

    def _decompose_query(self, state: ArxivResearcherState) -> dict:
        goal: Goal = state.goal
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(goal.content)
        return {
            "tasks": decomposed_tasks.tasks,
            "current_task_index": 0,
            "results": [],
        }

    def _execute_task(self, state: ArxivResearcherState) -> dict:
        results: list[dict] = self.task_executor.run(
            goal_setting=state.goal.content,
            tasks=state.tasks,
        )
        return {
            "results": results,
        }

    def _route_execute_task(
        self, state: ArxivResearcherState
    ) -> Literal["generate_report", "terminate"]:
        if len(state.results) == 0:
            return "terminate"
        else:
            return "generate_report"

    def _generate_report(self, state: ArxivResearcherState) -> dict:
        results: list[dict] = state.results
        query: str = state.goal.content
        final_output: str = self.reporter.run(
            context="\n".join([dict_to_xml_str(item) for item in results]),
            query=query,
        )
        return {"final_output": final_output}

    def _terminate(self, state: ArxivResearcherState) -> None:
        self.goal_optimizer.reset()
        pass

    def _stream_events(self, human_message: str | None, thread_id: str):
        if human_message:
            self._notify(
                "chat_message",
                ChatMessage(role="user", content=human_message),
            )
        for event in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            # 実行ノードの情報を取得
            node = list(event.keys())[0]
            if node in [
                "goal_setting",
                "decompose_query",
                "execute_task",
                "generate_report",
                "terminate",
            ]:
                if node == "goal_setting":
                    self._goal_setting_message(event[node])
                elif node == "decompose_query":
                    self._decompose_query_message(event[node])
                elif node == "execute_task":
                    self._execute_task_message(event[node])
                elif node == "generate_report":
                    self._generate_report_message(event[node])
                elif node == "terminate":
                    self._terminate_message(event[node])
                else:
                    pass

    def _goal_setting_message(self, update_state: dict) -> None:
        goal: Goal = update_state["goal"]
        if goal.is_need_human_feedback:
            message = ChatMessage(
                role="assistant",
                content=f"【追加質問】\n{goal.additional_question}",
            )
        else:
            message = ChatMessage(
                role="assistant", content=f"【目標設定】\n{goal.content}"
            )
        self._notify("chat_message", message)

    def _decompose_query_message(self, update_state: dict) -> None:
        tasks = "\n".join([f"- {task}" for task in update_state["tasks"]])
        message = ChatMessage(
            role="assistant", content=f"タスクを分解しました。\n{tasks}"
        )
        self._notify("chat_message", message)

    def _execute_task_message(self, update_state: dict) -> None:
        results: list[dict] = update_state["results"]
        if len(results) == 0:
            message = ChatMessage(
                role="assistant",
                content="検索結果が見つかりませんでした。",
            )
            self._notify("chat_message", message)
        else:
            data = [item for item in results]
            message = DataframeMessage(
                role="assistant",
                content=f"参考になる文献が{len(results)}件見つかりました。",
                data=data,
            )
            self._notify("dataframe_message", message)

    def _generate_report_message(self, update_state: dict) -> None:
        final_output: str = update_state["final_output"]
        message = ChatMessage(
            role="assistant",
            content=final_output,
        )
        self._notify("chat_message", message)

    def _terminate_message(self, update_state: dict) -> None:
        message = AlertMessage(
            role="assistant",
            content="調査が終了しました。新しい文献調査を行う場合は、続けて質問してください。",
        )
        self._notify("alert_message", message)
