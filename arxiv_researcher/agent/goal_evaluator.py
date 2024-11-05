from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

EVALUATOR_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<context>
{context}
</context>

<goal_setting>
{goal_setting}
</goal_setting>

<task>
goal_settingタグに記述された内容を実現するため、contextタグの内容を収集しました。収集した情報を基に、goal_settingタグに記述された内容を見直す必要があるかどうかを評価してください。
</task>

<hint>
goal_settingタグの内容は事前情報がない状態で生成された可能性があります。新しい情報を基に目標を見直すことで、ユーザーにとってより良い結果が得られる可能性があります。
</hint>
""".strip()


class GoalEvaluation(BaseModel):
    is_reset: bool = Field(
        default=False,
        description="目標を見直す必要がある場合はTrue",
    )
    reason: str = Field(
        description="目標を見直す必要がある理由を日本語で端的に表す",
    )
    content: str = Field(
        description="評価結果を踏まえて、改めてレポート生成のためのクエリを日本語で表す",
    )


class GoalEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, context: str, goal_setting: str) -> GoalEvaluation:
        prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(GoalEvaluation)
        return chain.invoke(
            {
                "current_date": self.current_date,
                "context": context,
                "goal_setting": goal_setting,
            }
        )
