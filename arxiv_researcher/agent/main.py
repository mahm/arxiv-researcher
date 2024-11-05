import operator
from logging import getLogger
from typing import Annotated, Callable, Iterator, Literal

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.types import StateSnapshot
from pydantic import BaseModel, Field

from arxiv_researcher.agent.event_emitter import EventEmitter
from arxiv_researcher.agent.goal_evaluator import GoalEvaluation, GoalEvaluator
from arxiv_researcher.agent.goal_optimizer import GoalOptimizer
from arxiv_researcher.agent.query_decomposer import DecomposedTasks, QueryDecomposer
from arxiv_researcher.agent.reporter import Reporter
from arxiv_researcher.agent.task_evaluator import TaskEvaluation, TaskEvaluator
from arxiv_researcher.agent.task_executor import TaskExecutor
from arxiv_researcher.agent.user_hearing import Hearing, HumanFeedbackChecker
from arxiv_researcher.agent.utility import dict_to_xml_str
from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
from arxiv_researcher.ui.types import (
    AlertMessage,
    ChatMessage,
    DataframeMessage,
    ExpandMessage,
    Message,
)

logger = getLogger(__name__)


class ArxivResearcherState(BaseModel):
    human_inputs: Annotated[list[str], operator.add] = Field(default_factory=list)
    hearing: Hearing = Field(default=None)
    goal: str = Field(default="")
    history: list[dict[str, str]] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    results: list[dict] = Field(default_factory=list)
    evaluation: GoalEvaluation | TaskEvaluation = Field(default=None)
    final_output: str = Field(default="")


class ArxivResearcher(EventEmitter):
    def __init__(self, llm: ChatOpenAI) -> None:
        super().__init__()
        self.subscribers: list[Callable[[str, Message], None]] = []
        self.user_hearing = HumanFeedbackChecker(llm)
        self.goal_optimizer = GoalOptimizer(llm)
        self.query_decomposer = QueryDecomposer(llm)
        self.task_executor = TaskExecutor(
            llm, searcher=ArxivSearcher(llm, event_emitter=self)
        )
        self.goal_evaluator = GoalEvaluator(llm)
        self.task_evaluator = TaskEvaluator(llm)
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

    def handle_human_message(self, query: str, thread_id: str) -> Iterator[Message]:
        logger.debug(f"メッセージ処理開始: query={query}, thread_id={thread_id}")
        node = (
            "human_feedback" if self.is_next_human_feedback_node(thread_id) else START
        )
        logger.debug(f"現在のノード: {node}")
        self.graph.update_state(
            config=self._config(thread_id),
            values={"human_inputs": [query]},
            as_node=node,
        )
        return self._stream_events(query=query, thread_id=thread_id)

    def is_next_human_feedback_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_feedback"

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()

    def reset(self) -> None:
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(ArxivResearcherState)
        graph.add_node("user_hearing", self._user_hearing)
        graph.add_node("human_feedback", self._human_feedback)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("evaluate_goal", self._evaluate_goal)
        graph.add_node("evaluate_task", self._evaluate_task)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("terminate", self._terminate)

        graph.add_edge(START, "user_hearing")
        graph.add_conditional_edges("user_hearing", self._route_user_hearing)
        graph.add_edge("human_feedback", "user_hearing")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_conditional_edges("execute_task", self._route_execute_task)
        graph.add_conditional_edges("evaluate_goal", self._route_evaluate_goal)
        graph.add_conditional_edges("evaluate_task", self._route_evaluate_task)
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
        return {"configurable": {"thread_id": thread_id, "recursion_limit": 1000}}

    def _user_hearing(self, state: ArxivResearcherState) -> dict:
        logger.info("hearing")
        human_input = state.human_inputs[-1]
        hearing = self.user_hearing.run(human_input, state.history)
        state.history.append({"role": "user", "content": human_input})
        if hearing.is_need_human_feedback:
            state.history.append(
                {"role": "assistant", "content": hearing.additional_question}
            )
        if len(state.history) > 10:
            state.history.pop(0)
        return {"hearing": hearing, "history": state.history}

    def _route_user_hearing(
        self, state: ArxivResearcherState
    ) -> Literal["human_feedback", "goal_setting"]:
        return (
            "human_feedback" if state.hearing.is_need_human_feedback else "goal_setting"
        )

    def _human_feedback(self, state: ArxivResearcherState) -> None:
        logger.info("human_feedback")
        # ユーザーからの入力を待つ
        pass

    def _goal_setting(self, state: ArxivResearcherState) -> dict:
        logger.info("goal_setting")
        mode = "conversation" if state.evaluation is None else "search"
        goal: str = self.goal_optimizer.run(
            history=state.history,
            mode=mode,
            search_results=state.results if mode == "search" else None,
            improvement_hint=(
                f"目標を見直す理由: {state.evaluation.reason}\n"
                f"評価結果を踏まえた新しいクエリ: {state.evaluation.content}"
                if state.evaluation
                else None
            ),
        )
        return {
            "goal": goal,
        }

    def _decompose_query(self, state: ArxivResearcherState) -> dict:
        logger.info("decompose_query")
        content = state.evaluation.content if state.evaluation else state.goal
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(content)
        return {
            "tasks": decomposed_tasks.tasks,
        }

    def _execute_task(self, state: ArxivResearcherState) -> dict:
        logger.info("execute_task")
        new_results: list[dict] = self.task_executor.run(
            user_hearing=state.goal,
            tasks=state.tasks,
        )
        results = [*state.results, *new_results]
        return {
            "results": results,
        }

    def _route_execute_task(
        self, state: ArxivResearcherState
    ) -> Literal["evaluate_goal", "evaluate_task", "terminate"]:
        if len(state.results) == 0:
            return "terminate"
        else:
            # NOTE: 既に評価結果がある場合はゴールを見直さない
            return "evaluate_goal" if state.evaluation is None else "evaluate_task"

    def _evaluate_goal(self, state: ArxivResearcherState) -> dict:
        logger.info("evaluate_goal")
        evaluation: GoalEvaluation = self.goal_evaluator.run(
            context="\n".join([dict_to_xml_str(item) for item in state.results]),
            goal_setting=state.goal,
        )
        return {
            "evaluation": evaluation,
        }

    def _route_evaluate_goal(
        self, state: ArxivResearcherState
    ) -> Literal["goal_setting", "evaluate_task"]:
        return "goal_setting" if state.evaluation.is_reset else "evaluate_task"

    def _evaluate_task(self, state: ArxivResearcherState) -> dict:
        logger.info("evaluate_task")
        evaluation: TaskEvaluation = self.task_evaluator.run(
            context="\n".join([dict_to_xml_str(item) for item in state.results]),
            goal_setting=state.goal,
        )
        return {
            "evaluation": evaluation,
        }

    def _route_evaluate_task(
        self, state: ArxivResearcherState
    ) -> Literal["decompose_query", "generate_report"]:
        return "decompose_query" if state.evaluation.is_reset else "generate_report"

    def _generate_report(self, state: ArxivResearcherState) -> dict:
        logger.info("generate_report")
        results: list[dict] = state.results
        query: str = state.goal
        final_output: str = self.reporter.run(
            context="\n".join([dict_to_xml_str(item) for item in results]),
            query=query,
        )
        return {
            "final_output": final_output,
        }

    def _terminate(self, state: ArxivResearcherState) -> dict:
        return {
            "human_inputs": [],
            "hearing": None,
            "goal": None,
            "history": [],
            "tasks": [],
            "results": [],
            "evaluation": None,
            "final_output": "",
        }

    def _stream_events(self, query: str | None, thread_id: str) -> Iterator[Message]:
        for event in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            # 実行ノードの情報を取得
            node = list(event.keys())[0]
            if node in [
                "user_hearing",
                "goal_setting",
                "decompose_query",
                "execute_task",
                "evaluate_goal",
                "evaluate_task",
                "generate_report",
                "terminate",
            ]:
                yield self._process_node_event(node, event[node])

    def _process_node_event(self, node: str, update_state: dict) -> Message:
        if node == "user_hearing":
            return self._user_hearing_message(update_state)
        elif node == "goal_setting":
            return self._goal_setting_message(update_state)
        elif node == "decompose_query":
            return self._decompose_query_message(update_state)
        elif node == "execute_task":
            return self._execute_task_message(update_state)
        elif node == "evaluate_goal":
            return self._evaluate_goal_message(update_state)
        elif node == "evaluate_task":
            return self._evaluate_task_message(update_state)
        elif node == "generate_report":
            return self._generate_report_message(update_state)
        elif node == "terminate":
            return self._terminate_message(update_state)

    def _user_hearing_message(self, update_state: dict) -> Message:
        hearing: Hearing = update_state["hearing"]
        if hearing.is_need_human_feedback:
            return Message(
                is_need_human_feedback=True,
                content=ChatMessage(
                    role="assistant",
                    content=hearing.additional_question,
                ),
            )
        else:
            return Message(
                content=ChatMessage(
                    role="assistant",
                    content="ご協力ありがとうございました。頂いた内容を基に文献調査を行います。",
                )
            )

    def _goal_setting_message(self, update_state: dict) -> Message:
        goal: str = update_state["goal"]
        return Message(
            content=ExpandMessage(
                role="assistant", title="ゴールを作成しました", content=goal
            )
        )

    def _decompose_query_message(self, update_state: dict) -> Message:
        tasks = "\n".join([f"- {task}" for task in update_state["tasks"]])
        return Message(
            content=ExpandMessage(
                role="assistant",
                title="調査タスクを分解しました",
                content=tasks,
            )
        )

    def _execute_task_message(self, update_state: dict) -> Message:
        results: list[dict] = update_state["results"]
        if not results:
            return Message(
                content=ChatMessage(
                    role="assistant", content="検索結果が見つかりませんでした。"
                )
            )
        return Message(
            content=ExpandMessage(
                role="assistant",
                title=f"参考になる文献が{len(results)}件見つかりました。",
                content=results,
            )
        )

    def _evaluate_goal_message(self, update_state: dict) -> Message:
        evaluation: GoalEvaluation = update_state["evaluation"]
        title = (
            "目標を見直す必要があります"
            if evaluation.is_reset
            else "目標は変更されません"
        )
        return Message(
            content=ExpandMessage(
                role="assistant",
                title=title,
                content=evaluation.reason,
            )
        )

    def _evaluate_task_message(self, update_state: dict) -> Message:
        evaluation: TaskEvaluation = update_state["evaluation"]
        title = (
            "追加の調査が必要です"
            if evaluation.is_reset
            else "十分な情報が集まりました"
        )

        return Message(
            content=ExpandMessage(
                role="assistant",
                title=title,
                content=evaluation.reason,
            )
        )

    def _generate_report_message(self, update_state: dict) -> Message:
        return Message(
            content=ChatMessage(role="assistant", content=update_state["final_output"])
        )

    def _terminate_message(self, update_state: dict) -> Message:
        return Message(
            is_done=True,
            content=AlertMessage(
                role="assistant",
                content="調査が終了しました。新しい文献調査を行う場合は、続けて質問してください。",
            ),
        )
