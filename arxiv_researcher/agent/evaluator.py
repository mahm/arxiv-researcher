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
goal_settingタグに記述された内容を実現するため、contextタグの内容を収集しました。収集された情報が十分かどうかを評価してください。
</task>

<rules>
1. 論文には参考文献があり、それらを読むことでより深く理解できる場合があります。
2. 最終レポートのためには具体的な実験結果が必要です。
</rules>
""".strip()


class Evaluation(BaseModel):
    need_more_information: bool = Field(
        default=False,
        description="最終レポートを生成するためにさらに情報が必要かどうか",
    )
    reason: str = Field(
        default="",
        description="評価の理由を日本語で端的に表す",
    )
    content: str = Field(
        default="",
        description="追加の調査として必要な内容を詳細に日本語で記述",
    )


class Evaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, context: str, goal_setting: str) -> Evaluation:
        prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(Evaluation)
        return chain.invoke(
            {
                "current_date": self.current_date,
                "context": context,
                "goal_setting": goal_setting,
            }
        )
