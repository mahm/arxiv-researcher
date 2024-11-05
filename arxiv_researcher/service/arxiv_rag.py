from tempfile import TemporaryDirectory
from typing import Iterable, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arxiv_researcher.service.pdf_to_markdown import PdfToMarkdown
from arxiv_researcher.settings import settings


class DoclingPDFLoader(BaseLoader):
    def __init__(self, url: str | list[str]) -> None:
        self._urls = url if isinstance(url, list) else [url]

    def lazy_load(self) -> Iterator[LangchainDocument]:
        for url in self._urls:
            converter = PdfToMarkdown(url)
            yield LangchainDocument(page_content=converter.convert())


class ArxivRAG:
    def __init__(self, pdf_url: str) -> None:
        loader = DoclingPDFLoader(pdf_url)
        embeddings = HuggingFaceEmbeddings(model_name=settings.hf_embeddings_model)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(loader.load())
        milvus_uri = f"{(tmp_dir := TemporaryDirectory()).name}/milvus.db"
        self.vectorstore = Milvus.from_documents(
            splits,
            embeddings,
            connection_args={"uri": milvus_uri},
            drop_old=True,
        )

    def _expand_query(self, query: str) -> str:
        prompt = PromptTemplate.from_template(
            """\
Please expand the following query into a focused academic search format.
Generate a concise and specific search query including:

- Core concepts and their synonyms
- Related technical terms and methodologies
- Key domain-specific keywords

Example:
- "Code Generation Datasets in Research and Industry"
- "Case Studies on Code Synthesis Applications"
- "Effectiveness of Automated Code Generation Data"
- "Practical Applications of Program Synthesis Datasets"
- "Impact Assessment of Code Generation in Software Development" 
            
Query: {query}

Expanded Query:"""
        )

        chain = prompt | settings.fast_llm | StrOutputParser()
        return chain.invoke({"query": query})

    def run(self, query: str, k: int = 30) -> str:
        def format_docs(docs: Iterable[LangchainDocument]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        prompt = PromptTemplate.from_template(
            """<context>
{context}
</context>

Please answer the following question using only the information within the context tags (without using prior knowledge).

Question: {question}

Please format your answer as follows:

- Answer
  - [Please quote relevant sections from the paper in detail. If quoting from multiple sections, separate each quote into paragraphs]
    - [Based on the above quotes, explain the content in an easy to understand way]
- Related Papers
  - [Please quote sections that mention related research or cited papers in detail, including specific paper titles, authors, and years when available (e.g. "GPT-3: Language Models are Few-Shot Learners" by Brown et al. 2020)]
  - For each significant paper mentioned:
    - Title and authors
    - Key contributions or findings discussed in the context
    - Why this paper is relevant to the current research
    - Whether further investigation of this paper is recommended based on its significance to the topic

Note: If no relevant content is found in the paper, please write "[NOT_RELATED]"
Note: Please ensure quotes are long enough to provide sufficient context."""
        )

        chain = (
            {
                "context": RunnablePassthrough()
                | self._expand_query
                | retriever
                | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | settings.fast_llm
            | StrOutputParser()
        )
        return chain.invoke(query)


if __name__ == "__main__":
    rag = ArxivRAG("https://arxiv.org/pdf/2206.01062")
    print(rag.run("What is the main idea of this paper?"))
