---
title: "ChatGPT API 사용하기"
categories: Project
toc: true
toc_sticky: true
---

어떤 천재적인 고등학생이 리버스엔지니어링을 통해 vscode에서 chatgpt를 이용할 수 있게 만들어 놓았다.  
사용방법은 엄청 쉽지만, 내가 헤메었던 부분을 위주로 작성하였다.  
원본 깃허브 레포는 [여기](https://github.com/acheong08/ChatGPT)로 가면 된다. 수시로 업데이트 되기 때문에 사용하기전 반드시 확인해야한다!


### 깃허브 레포지토리 클론
`git clone https://github.com/acheong08/ChatGPT.git` 명령어를 통해 깃허브 레포를 클론한다.

### OpenAI Auth
OpenAI의 인증 시스템이 필요하다.  
OpenAI의 이메일주소, 비밀번호와 세션 토큰과 엑세스 토큰이 필요하다.  
먼저 세션 토큰을 얻는 방법은 http://chat.openai.com 여기에 들어가서 `F12` 키를 눌러 DevTool로 들어간다. 그 중 Application이라고 써있는 항목에 들어가면 목록에 `__Secure-next-auth.session-token` 이라는 이름이 있는데, 이 값이 세션 토큰이다. 복사에서 config에 붙여넣자.  

![png](assets/images/Project/session_token.png)  

그 다음은 엑세스 토큰을 얻어야 하는데, 이는 https://chat.openai.com/api/auth/session 여기로 들어가면 "accessToken"의 값이 있는데, 이를 넣어주면 된다.

### 코드 작성
위에서 얻은 이메일과, 패스워드, 세션토큰과, 엑세스 토큰을 입력한 뒤에 prompt에 질문을 넣으면 된다.  


```python
from revChatGPT.V1 import Chatbot

chatbot = Chatbot(config={
  "email": "",
  "password": "",
  "session_token":"",
  "access_token": "<>"
})

prompt = "how many beaches does portugal have?"
response = ""
prev_text = "" 

for data in chatbot.ask(
  prompt,
  conversation_id=chatbot.config.get("conversation"),
  parent_id=chatbot.config.get("parent_id"),
):
    response += data["message"][len(prev_text) :]
    prev_text = data["message"] 

print(response) 
```

    Portugal is a country with a long coastline, so it has many beaches. The exact number of beaches in Portugal is difficult to determine, but it is estimated that there are over 900 beaches along the mainland coast and islands. Some of the most famous beaches in Portugal include Praia da Rocha, Albufeira, Nazaré, Cascais, and Funchal.
    

### 주의사항
수시로 업데이트 되니 본 레포지토리를 참고하자. 내가 사용할 때는 거의 하루에 한번씩 업데이트 되었었다...