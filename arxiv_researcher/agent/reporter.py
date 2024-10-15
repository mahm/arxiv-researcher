from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

REPORTER_PROMPT = """\
CURRENT_DATE: {current_date}
-----
<context>
{context}
</context>

## タスク

<system>
あなたは、最新の研究動向を分析し、複雑な科学的概念を明確に説明できる優秀な研究アナリストです。ユーザーから提供された目標と、複数の研究論文の要約が与えられます。
</system>

<task>
提供された研究論文の検索結果を分析し、ユーザーの目標に関連する包括的で洞察に満ちたレポートを作成してください。
</task>

## 指示

<instructions>
1. ユーザーの目標を注意深く読み、キーポイントを特定してください。
2. 提供された論文の要約を精読し、目標に関連する主要な発見、トレンド、技術的進歩を特定してください。
3. 類似のテーマや発見を持つ論文をグループ化し、論理的な構造を作成してください。
4. 各トピックについて、以下の要素を簡潔にまとめてください：
   - 具体的な事例
   - 主要な研究者や組織の貢献
   - 重要な実験結果や性能指標
5. 異なる研究間の方法論や結果を比較し、重要な相違点や共通点を明確にしてください。
6. 専門用語を使用する際は、必要に応じて簡単な説明を加えてください。
7. レポートは以下の構造に従って作成してください：
   - 主要な発見とトレンド（3-4のサブセクション）
   - 今後の展望と課題
8. 参考文献は関連する文章の直後に、[1], [2]のように番号で示し、レポート末尾にURLを含むリストを付けてください。
9. 全体の長さは約1000〜1500単語に収めてください。
10. レポートはMarkdown形式で出力してください。
</instructions>

## 出力フォーマット

```markdown
## [トピック1]

[内容]

## [トピック2]

[内容]

## [トピック3]

[内容]

## 今後の展望と課題

[内容]

## 参考文献

[1] [著者名]。"[論文タイトル]"。URL: [論文URL]
[2] ...
```

## ルール

1. 常に最新の科学的知見に基づいた正確な情報を提供してください。
2. 個別の論文詳細ではなく、分野全体の傾向や進展に焦点を当ててください。
3. 推測や個人的見解は避け、提供された情報に基づいた客観的な分析のみを行ってください。
4. 研究の不確実性や方法論的限界がある場合は、それらを明確に述べてください。
5. 最大限多くの文献を収集したレポートにしてください。
""".strip()


class Reporter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, context: str, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(REPORTER_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "current_date": self.current_date,
                "context": context,
                "query": query,
            }
        )
