---
title: "랭체인으로 PDF 문서를 읽고 대화내용을 기억하는 LLM 만들기"
categories: Project
toc: true
toc_sticky: true
---
본 포스팅은 `langchain==0.0.229`를 사용합니다. 현재(2024/04/15)기준 최신 버전은`langchain==0.1.16`이므로 참고 바랍니다.  
또한 `gpt-3.5-turbo-0125`모델을 사용하므로 openai api키가 있어야합니다.

### API키 설정
```python
from dotenv import load_dotenv
import os

load_dotenv()
openApiKey = os.environ.get('OPENAI_API_KEY')
```

### PDF불러오기
```python
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

# loader = PyPDFLoader(path)
loader = PyPDFDirectoryLoader("./test")
```
`PyPDFLoader()`는 PDF파일이 1개일 때, `PyPDFDirectoryLoader()`는 디렉토리 안에 PDF가 있을 때 사용한다.  
디렉토리 안에 PDF가 한개 있든 여러개 있든 상관없다.

### PDF파싱하기
```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)

data = loader.load_and_split()
docsearch = FAISS.from_documents(data, embeddings)
```
위에서 load한 pdf를 split하고 FAISS를 이용하여 텍스트를 임베딩하고 벡터 공간에 매핑시킨다.  
임베딩은 OpenAIEmbedding을 사용한다.

### 저장 및 로드
```python

docsearch.save_local("./vec")
docsearch = FAISS.load_local("./vec", embeddings)
```
pdf문서를 벡터화 하는 작업은 한번만 하면 된다.(한번 할 때 시간이 꽤 걸린다. 그래서 한번 하고 저장한 것을 계속 불러다 사용하면 됨.)


### 프롬프트 설정
```python
from langchain.prompts import PromptTemplate

template = """
당신은 동화에 대한 궁금증을 답변하는 챗봇입니다. 
데이터(<ctx></ctx> 로 구분)와 채팅 기록(<hs></hs> 로 구분)을 사용하여 질문에 대답하세요.
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
질문: {question}
"""

prompt = PromptTemplate(input_variables=["context", "question", "history"], template=template)
```
`{context}`에는 질문에 관련된 pdf내용이 들어가고, `{history}`에는 지금까지 대화한 내용이 들어간다.  
`{history}`를 넣어줌으로써 웹에서 사용할 때와 같이 gpt가 이전내용을 기억한다.  
> 프롬프트 템플릿 종류가 많으니 다른 것들은 랭체인 공식 문서를 참고하기 바란다.

### LLM설정
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0, openai_api_key=openApiKey, max_tokens=400)
```
LLM 모델로는 gpt-3.5를 사용한다. 이것도 마찬가지로 사용할 수 있는 LLM 종류가 많으니 공식 문서를 참고하자.

### 문서 벡터 설정
```python
retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'lambda_mult': 0.8})
```
이 부분은 관련 PDF 내용을 검색할 때 어떤 형식으로 검색할 지 정하는 부분이다.  
위 함수는 "검색방식은 mmr(Maximum marginal relevance search)로 하고 점수가 0.8 이상인 부분만 참고할게요" 라는 뜻이다.  
mmr말고 다른 방식도 있다.(similarity, similarity_score_threshold 등등..)

### 대화 메모리 설정
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(memory_key="history", input_key="question", k=3)
```
이 부분은 대화 내용을 어디까지 기억하는지 설정한다. k=3이라면 3번째 대화까지만 기억한다.  
메모리 부분도 엄청 많으니 공식 문서를 참고해서 원하는 메모리 버퍼를 사용하자.

### 체인 설정
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        chain_type_kwargs={
                                            "prompt": prompt,
                                            "verbose": False,
                                            "memory": memory,
                                        }
                                        )
```
이 부분은 지금까지 우리가 만든 컴포넌트(?)들을 하나로 합치는 부분이다.  
이제 이 chain을 호출해 주기만 하면 알아서 문서를 찾고, 대화 기억하고 등등 한방에 작업이 이루어 진다.  
체인 종류도 어마무시하게 많다..

### 작동
```python
print(qa_chain("안녕하세요"))
```
```text
{'query': '안녕하세요', 'result': '안녕하세요! 동화에 대한 궁금증이 있으신가요? 무엇을 도와드릴까요?'}
```

### 테스트
우선 필자는 `"./test"` 폴더 안에 단군, 토끼와거북이, 흥부와 놀부의 내용이 담긴 PDF를 넣어놓고, `"./vec"`폴더에 벡터를 저장시켰다.  
```python
print(qa_chain("단군신화에서 곰과 호랑이는 동굴에서 무엇을 먹었어?"))
print(qa_chain("흥부와 놀부에서 나쁜사람이 누구야?"))
print(qa_chain("제비가 가져온 씨앗은 무슨 씨앗이야?"))
print(qa_chain("토끼와 거북이 중에 누가 먼저 산꼭대기에 도착했어?"))
```
```text
{'query': '단군신화에서 곰과 호랑이는 동굴에서 무엇을 먹었어?', 'result': '단군신화에서 곰과 호랑이는 동굴에서 쑥과 마늘을 먹었습니다.'}
{'query': '흥부와 놀부에서 나쁜사람이 누구야?', 'result': '놀부가 나쁜 사람입니다.'}
{'query': '제비가 가져온 씨앗은 무슨 씨앗이야?', 'result': '답변: 제비가 가져온 씨앗은 "박" 씨앗입니다.'}
{'query': '토끼와 거북이 중에 누가 먼저 산꼭대기에 도착했어?', 'result': '거북이가 먼저 산꼭대기에 도착했습니다.'}
```
결과가 꽤 정확하게 나오는것 같다!! 하지만 gpt가 지어낸건지 pdf에서 가져와서 답변하는 건지 확인해 보도록 하자.

```python
source_documents = retriever.get_relevant_documents("단군신화에서 곰과 호랑이는 동굴에서 무엇을 먹었어?")[0]
print(source_documents)
```
`get_relevant_documents()`는 사용자의 질문이랑 유사한 내용을 가져오는 함수이다. 기본적으로 k=5이기 때문에 가장 유사한 첫 번째 문서만 가져와봤다.
```text
page_content='8   그러자 환웅이 대답했어요. \n“백일 동안  동굴에서  쑥과  마늘만 먹어야 한다.\n   그러면 사람이 될 것이다.”\n      곰과     호랑이는 기뻤어요.  \n      곰과     호랑이는     쑥과     마늘을 가지고\n      동굴로 들어갔어요.\n      곰과     호랑이는 기뻤어요.  \n      곰과     호랑이는     쑥과     마늘을 가지고\n      곰과     호랑이는 기뻤어요.  \n      곰과     호랑이는     쑥과     마늘을 가지고\n      곰과     호랑이는     쑥과     마늘을 가지고\n      곰과     호랑이는     쑥과     마늘을 가지고\n      동굴로 들어갔어요.\n한국어 4-22 단군이야기.indd   8 2014-11-10   오후 2:28:24' metadata={'source': 'test\\단군.pdf', 'page': 8}
```
PDF자체가 그림안에 글이 있다보니까 반복되는 부분을 가져온거 같지만 관련 내용을 정확하게 가져오는것을 알 수 있다!

참고로 qa_chain에서 `verbose=True`를 하면 템플릿 안에 값이 어떻게 들어가는지 보여준다.  
반복문을 돌려서 qa_chain이 하나의 프로세스에서 동작하면 히스토리 값도 출력이 된다!

```text
당신은 동화에 대한 궁금증을 답변하는 챗봇입니다. 
데이터(<ctx></ctx> 로 구분)와 채팅 기록(<hs></hs> 로 구분)을 사용하여 질문에 대답하세요.
<ctx>
11       토끼가 뛰어가다가 뒤를 돌아보았어요. 
“       거북이가 어디에 있지 ?”
       거북이는 멀리서 엉금엉금 기어오고 있었어요.
“       거북이가 어디에 있지 ?”
       거북이는 멀리서 엉금엉금 기어오고 있었어요.
       토끼가 뛰어가다가 뒤를 돌아보았어요. 
한국어 2-8 토끼와 거북이.indd   11 2014-05-08   오후 4:01:14

7
        토끼가 말했어요. 
“좋아. 그럼 우리 달리기 경주를 하자. 산꼭대기까지        
   먼저 가면 이기는 거다.” 
        토끼가 말했어요. 
한국어 2-8 토끼와 거북이.indd   7 2014-05-08   오후 4:01:06

2
2
옛날 옛날에      토끼와       거북이가 있었어요. 
      토끼는 빨랐어요. 아주 빨랐어요. 
옛날 옛날에      토끼와       거북이가 있었어요. 
      토끼는 빨랐어요. 아주 빨랐어요. 
옛날 옛날에      토끼와       거북이가 있었어요. 
한국어 2-8 토끼와 거북이.indd   2 2014-05-08   오후 4:00:58

17
         거북이는 쉬지 않고 계속 걸어 산꼭대기에 도착했어요.             
         거북이가 소리쳤어요.
“내가 이겼다. 만세 !”
   그 소리에      토끼는 잠에서 깼어요. 
         거북이는 쉬지 않고 계속 걸어 산꼭대기에 도착했어요.             
         거북이가 소리쳤어요.
   그 소리에      토끼는 잠에서 깼어요. 
한국어 2-8 토끼와 거북이.indd   17 2014-05-08   오후 4:01:26
</ctx>
------
<hs>
Human: 단군신화에서 곰과 호랑이는 동굴에서 무엇을 먹었어?
AI: 단군신화에서 곰과 호랑이는 동굴에서 쑥과 마늘을 먹었습니다.
Human: 흥부와 놀부에서 나쁜사람이 누구야?
AI: 답변: 흥부와 놀부에서 나쁜 사람은 놀부입니다.
Human: 제비가 가져온 씨앗은 무슨 씨앗이야?
AI: 답변: 제비가 가져온 씨앗은 박 씨앗입니다.
</hs>
질문: 토끼와 거북이 중에 누가 먼저 산꼭대기에 도착했어?
```

### 전체 코드
```python

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

load_dotenv()
openApiKey = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)

# loader = PyPDFDirectoryLoader("./test")
# data = loader.load_and_split()
# docsearch = FAISS.from_documents(data, embeddings)
# docsearch.save_local("./vec")

docsearch = FAISS.load_local("./vec", embeddings)

template = """
당신은 동화에 대한 궁금증을 답변하는 챗봇입니다. 
데이터(<ctx></ctx> 로 구분)와 채팅 기록(<hs></hs> 로 구분)을 사용하여 질문에 대답하세요.
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
질문: {question}
"""

prompt = PromptTemplate(input_variables=["context", "question", "history"], template=template)
llm = ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0, openai_api_key=openApiKey, max_tokens=400)
retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'lambda_mult': 0.8})
memory = ConversationBufferWindowMemory(memory_key="history", input_key="question", k=3)
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        chain_type_kwargs={
                                            "prompt": prompt,
                                            "verbose": True,
                                            "memory": memory,
                                        }
                                        )


for i in range(4):
    input_text = input("질문을 입력하세요: ")
    result = qa_chain(input_text)
    print(result)
```

### 결론
정확한 답변을 얻으려면 PDF도 전처리가 되어있으면 좋다!  
PDF말고도 db나 text 등등을 읽을 수 있는 함수가 있으니 공식문서를 참고하자. 
