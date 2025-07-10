from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== 1단계: 문서 로딩 =====
# RAG 시스템의 첫 번째 단계는 지식 베이스가 될 문서들을 로드하는 것입니다.
# 여기서는 PDF 파일을 주로 사용하지만, 텍스트 파일도 추가할 수 있습니다.
documents = []

# PDF 파일 로드 - 실제 운영 환경에서는 여러 PDF 파일을 반복문으로 처리할 수 있습니다
pdf_path = "C:/Users/wclee/rag_env/sap_ai_core.pdf"
try:
    # PyPDFLoader는 PDF의 각 페이지를 별도 Document 객체로 변환합니다
    # 각 페이지는 page_content(내용)와 metadata(페이지 번호, 파일 경로 등)를 포함합니다
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
except Exception as e:
    print(f"Error loading PDF: {e}")

# 텍스트 파일도 함께 로드하고 싶다면 아래 코드의 주석을 해제하세요
# text_path = "path/to/your/document.txt"
# try:
#     loader = TextLoader(text_path)
#     documents.extend(loader.load())
# except Exception as e:
#     print(f"Error loading text file: {e}")

# ===== 2단계: 문서 분할 (Chunking) =====
# 큰 문서를 작은 청크로 나누는 이유:
# 1. LLM의 컨텍스트 윈도우 제한 때문에 너무 긴 텍스트를 한 번에 처리할 수 없음
# 2. 벡터 검색 시 더 정확한 매칭을 위해 의미적으로 관련된 작은 단위로 나눔
# 3. 검색 품질 향상 - 관련성 높은 부분만 정확히 찾아낼 수 있음
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크의 최대 글자 수 (한국어 기준 약 500-700자 정도)
    chunk_overlap=200     # 인접한 청크 간의 중복 글자 수 (문맥 연결성 유지를 위함)
)
# split_documents는 Document 객체 리스트를 받아서 더 작은 Document 객체 리스트로 반환
texts = text_splitter.split_documents(documents)
print(f"분할된 문서 청크 수: {len(texts)}")

# ===== 3단계: 임베딩 모델 초기화 =====
# 임베딩은 텍스트를 벡터(숫자 배열)로 변환하는 과정입니다
# 의미적으로 유사한 텍스트는 유사한 벡터 값을 가지게 됩니다
from langchain_community.embeddings import OllamaEmbeddings

# Ollama 로컬 서버에서 실행되는 임베딩 모델을 사용
# nomic-embed-text는 텍스트 임베딩에 특화된 모델입니다
# 로컬에서 실행되므로 데이터가 외부로 전송되지 않아 보안성이 좋습니다
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ===== 4단계: 벡터 데이터베이스 설정 =====
# 벡터 데이터베이스는 임베딩된 벡터들을 저장하고 유사도 검색을 수행합니다
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant

collection_name = "my_rag_collection"

# Qdrant는 고성능 벡터 검색을 위한 데이터베이스입니다
# 코사인 유사도 등을 사용해 질의와 가장 유사한 문서를 빠르게 찾아줍니다
qdrant_vectorstore = Qdrant.from_documents(
    texts,                              # 분할된 문서 청크들
    ollama_embeddings,                  # 임베딩 모델
    collection_name=collection_name,    # 컬렉션 이름 (테이블명 같은 개념)
    url="http://localhost:6333",        # Qdrant 서버 주소
    force_recreate=True                 # 기존 컬렉션이 있으면 삭제 후 새로 생성
)

print(f"Qdrant 컬렉션 '{collection_name}'에 {len(texts)}개의 문서가 저장되었습니다.")

# ===== 5단계: LLM 초기화 =====
# 실제 답변을 생성할 대형 언어 모델을 초기화합니다
from langchain_community.llms import Ollama

# Llama3.1 모델을 사용 - 로컬에서 실행되는 오픈소스 모델
# 상용 API 대비 비용이 들지 않고 데이터 보안이 좋습니다
ollama_llm = Ollama(model="llama3.1")

# ===== 6단계: 기본 RAG 시스템 구성 =====
# 리트리버는 질문에 관련된 문서들을 벡터 데이터베이스에서 검색하는 역할
# k=5는 상위 5개의 가장 유사한 문서를 가져온다는 의미
retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})

from langchain.chains import RetrievalQA

# RetrievalQA는 "검색 후 답변 생성"의 전체 파이프라인을 관리합니다
# 동작 과정: 질문 → 벡터 검색 → 관련 문서 검색 → LLM에 문서+질문 전달 → 답변 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,                     # 답변 생성용 LLM
    chain_type="stuff",                 # 검색된 문서들을 모두 컨텍스트에 포함시키는 방식
    retriever=retriever,                # 문서 검색기
    return_source_documents=True        # 답변과 함께 참조한 원본 문서도 함께 반환
)

# ===== 7단계: 기본 RAG 테스트 =====
# 리랭킹 적용 전의 기본 RAG 성능을 확인해봅시다
query = "SAP AI Core에서 모델을 학습시키는 기능이 있나요?"
result = qa_chain.invoke({"query": query})

print("--- RAG 답변 ---")
print(f"질문: {query}")
print(f"답변: {result['result']}")
print("\n--- 참조 문서 ---")
for doc in result['source_documents']:
    print(f"페이지: {doc.metadata.get('page')}, 소스: {doc.metadata.get('source')}")
    print(f"내용: {doc.page_content[:200]}...")
    print("-" * 20)

# ===== 8단계: 리랭킹 모델 준비 =====
# 리랭킹이 필요한 이유:
# 1. 벡터 검색만으로는 정확도가 제한적일 수 있음
# 2. 초기 검색 결과를 더 정교하게 순위를 매겨 답변 품질 향상
# 3. 의미적 유사성뿐만 아니라 실제 관련성까지 고려한 재정렬
from sentence_transformers import CrossEncoder

# CrossEncoder는 쿼리와 문서 쌍을 입력받아 관련성 점수를 직접 계산합니다
# TinyBERT 모델은 속도와 성능의 균형이 좋은 선택입니다
reranker_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')

def rerank_documents(query, documents, reranker, top_k=3):
    """
    검색된 문서들을 쿼리와의 관련성에 따라 재정렬하는 함수
    
    Args:
        query: 사용자 질문
        documents: 초기 검색으로 얻은 문서 리스트
        reranker: 리랭킹 모델 (CrossEncoder)
        top_k: 최종적으로 반환할 상위 문서 개수
    
    Returns:
        관련성 점수 순으로 재정렬된 상위 k개 문서
    """
    # 각 문서와 쿼리를 쌍으로 만들어 리랭킹 모델에 입력할 형태로 준비
    pairs = [(query, doc.page_content) for doc in documents]
    
    # 리랭킹 모델이 각 쌍의 관련성 점수를 계산 (높을수록 관련성이 높음)
    scores = reranker.predict(pairs)
    
    # 문서와 점수를 묶어서 점수 기준으로 내림차순 정렬
    scored_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # 상위 k개 문서만 반환 (점수는 제외하고 문서만)
    return [doc for doc, score in scored_documents[:top_k]]

# ===== 9단계: 리랭킹 컴프레서 구현 =====
# LangChain의 ContextualCompressionRetriever와 호환되는 커스텀 컴프레서를 만듭니다
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from typing import List, Any
from pydantic import Field

class RerankCompressor(BaseDocumentCompressor):
    """
    문서 리랭킹을 수행하는 커스텀 컴프레서
    
    BaseDocumentCompressor를 상속받아 LangChain 생태계와 호환됩니다.
    Pydantic 모델이므로 필드를 Field()로 정의해야 합니다.
    """
    
    # Pydantic 필드 정의 - 일반적인 클래스 변수 선언과 다름에 주의!
    reranker_model: Any = Field(description="리랭킹을 위한 CrossEncoder 모델")
    top_n: int = Field(default=3, description="최종 반환할 상위 문서 개수")
    
    class Config:
        # CrossEncoder 같은 외부 객체를 필드로 사용할 수 있도록 허용
        arbitrary_types_allowed = True

    def compress_documents(self, documents: List[Any], query: str, callbacks=None) -> List[Any]:
        """
        문서 압축(실제로는 리랭킹) 수행
        
        Args:
            documents: 초기 검색으로 얻은 문서 리스트
            query: 사용자 질문
            callbacks: 콜백 함수들 (보통 사용 안함)
        
        Returns:
            리랭킹된 상위 문서들
        """
        if not documents:
            return []
        
        # 위에서 정의한 리랭킹 함수를 호출하여 문서 재정렬
        reranked_docs = rerank_documents(query, documents, self.reranker_model, self.top_n)
        return reranked_docs

# ===== 10단계: 리랭킹 컴프레서 초기화 =====
# 키워드 인자를 명시적으로 사용하여 Pydantic 모델을 올바르게 초기화
rerank_compressor = RerankCompressor(
    reranker_model=reranker_model,  # 앞서 로드한 CrossEncoder 모델
    top_n=3                         # 최종적으로 3개의 문서만 LLM에 전달
)

# ===== 11단계: 리랭킹이 적용된 리트리버 구성 =====
# ContextualCompressionRetriever는 기본 리트리버와 컴프레서를 결합합니다
# 동작 과정: 질문 → 기본 검색(k=5) → 리랭킹 컴프레서 적용 → 상위 3개 문서 반환
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,           # 기본 벡터 검색기 (5개 문서 검색)
    base_compressor=rerank_compressor   # 리랭킹 컴프레서 (3개로 압축)
)

# ===== 12단계: 리랭킹이 적용된 QA 체인 구성 =====
# 이제 전체 파이프라인이 완성되었습니다:
# 질문 → 벡터 검색(5개) → 리랭킹(3개) → LLM 답변 생성
qa_rerank_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,                     # 동일한 LLM 사용
    chain_type="stuff",                 # 선택된 문서들을 모두 컨텍스트에 포함
    retriever=compression_retriever,    # 리랭킹이 적용된 리트리버
    return_source_documents=True        # 참조 문서 정보도 함께 반환
)

# ===== 13단계: 리랭킹 적용 결과 테스트 =====
# 동일한 질문으로 리랭킹 전후 결과를 비교해볼 수 있습니다
query_rerank = "SAP AI Core에서 모델을 학습시키는 기능이 있나요?"
result_rerank = qa_rerank_chain.invoke({"query": query_rerank})

print("\n--- RAG (Rerank 적용) 답변 ---")
print(f"질문: {query_rerank}")
print(f"답변: {result_rerank['result']}")
print("\n--- 참조 문서 (Reranked) ---")
for doc in result_rerank['source_documents']:
    print(f"페이지: {doc.metadata.get('page')}, 소스: {doc.metadata.get('source')}")
    print(f"내용: {doc.page_content[:200]}...")
    print("-" * 20)

# ===== 추가 팁 =====
# 1. 성능 비교: 기본 RAG와 리랭킹 적용 RAG의 답변을 비교해보세요
# 2. 파라미터 튜닝: chunk_size, chunk_overlap, k값, top_n 등을 조정해가며 최적화
# 3. 평가 지표: 실제 운영에서는 정확도, 관련성, 응답 시간 등을 측정하여 성능 평가
# 4. 모델 선택: 리랭킹 모델은 도메인에 따라 다른 모델을 선택할 수 있습니다