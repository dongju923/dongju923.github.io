---
title: "파이썬 코드로 웹과 똑같은 클로드 무료로 사용하기"
categories: Project
toc: true
toc_sticky: true
---
# `requests` 모듈을 이용한 Claude 무료 사용법

<div style="background-color:#fff3cd;border-left:6px solid #ff4d4d;padding:14px 18px;border-radius:6px;margin-bottom:20px;">
<strong style="color:#cc0000;font-size:1.05em;">⚠️ 경고</strong><br><br>
<span style="color:#7a0000;">
<code>requests</code> 모듈을 이용해 세션 정보를 추출하여 Claude를 사용하는 행위는 <a href="https://www.anthropic.com/legal/consumer-terms" style="color:#cc0000;">Anthropic 이용약관</a>에 위반될 수 있습니다.
사용으로 인해 발생하는 계정 정지 등의 불이익은 전적으로 사용자 본인의 책임입니다.
</span>
</div>

브라우저에서 추출한 세션 정보를 이용해 Claude API에 직접 요청을 보내는 방법을 안내합니다.

---

## 사전 준비

- [Claude](https://claude.ai/new) 계정 (무료 플랜 가능)
- Python에 `requests`, `json` 패키지 설치
- 개발자 도구(DevTools)를 지원하는 브라우저

---

## 1단계 — 대화 시작

[claude.ai/new](https://claude.ai/new)에 접속하여 새 대화를 시작합니다.

---

## 2단계 — 개발자 도구 열기

`F12`를 눌러 브라우저 개발자 도구를 엽니다.

---

## 3단계 — `completion` 요청 찾기

**Network → completion → Headers** 순서로 이동합니다.


> **참고:** `completion`이 보이지 않는다면 **Preserve Log**를 체크한 후, 현재 세션에서 아무 메시지나 전송하면 나타납니다.

---

## 4단계 — 필요한 값 추출

개발자 도구 패널에서 아래 네 가지 정보를 수집해야 합니다.

### 4.1 — Request URL 파라미터

**Headers → General → Request URL**에서 아래 두 값을 추출합니다.

- `lastActiveOrg` — 조직 ID
- `chat_session` — 현재 대화 세션 ID

![DevTools Network Headers](/assets/images/project_img/image.png)    

URL 형식은 다음과 같습니다:

```
https://claude.ai/api/organizations/{lastActiveOrg}/chat_conversations/{chat_session}/completion
```

### 4.2 — 세션 키 (Cookie)

**Headers → Request Headers → cookie**에서 `sessionKey=` 뒤에 있는 값을 복사합니다.

### 4.3 — 디바이스 ID

**Headers → Request Headers → anthropic-device-id**의 값을 복사합니다.

### 4.4 — User Agent

**Headers → Request Headers → user-agent**의 값을 복사합니다.

---

## 5단계 — 요청 구성

추출한 값을 아래 코드에 채워넣습니다.

### Cookies

```python
url = f"https://claude.ai/api/organizations/{lastActiveOrg}/chat_conversations/{chat_session}/completion"

cookies = {
    "sessionKey": f"{sessionKey}",
    "lastActiveOrg": f"{lastActiveOrg}",
    "anthropic-device-id": f"{anthropic_device_id}",
}
```

### Headers

```python
headers = {
    "Content-Type": "application/json",
    "accept": "text/event-stream",
    "anthropic-client-platform": "web_claude_ai",
    "anthropic-device-id": f"{anthropic_device_id}",
    "origin": "https://claude.ai",
    "referer": f"https://claude.ai/chat/{chat_session}",
    "user-agent": f"{user_agent}",
}
```

### Payload

**Network → completion → Payload**에서 값을 확인하고 아래 코드를 상황에 맞게 수정합니다. `prompt`에는 Claude에게 보낼 질문을 입력합니다.

![DevTools Payload](/assets/images/project_img/image2.png)

```python
payload = {
    "prompt": f"{prompt}",
    "timezone": "Asia/Seoul",
    "personalized_styles": [
        {
            "type": "default",
            "key": "Default",
            "name": "Normal",
            "nameKey": "normal_style_name",
            "prompt": "Normal\n",
            "summary": "Default responses from Claude",
            "summaryKey": "normal_style_summary",
            "isDefault": True
        }
    ],
    "locale": "ko-KR",
    "model": "claude-sonnet-4-6"
}
```

---

## 6단계 — 요청 전송 및 응답 수신

```python
import requests
import json

response = requests.post(
    url,
    headers=headers,
    cookies=cookies,
    json=payload,
    stream=True  # 스트리밍 응답 활성화
)

print(f"Status Code: {response.status_code}")

full_text = ""

for line in response.iter_lines():
    if line:
        decoded = line.decode("utf-8")

        if decoded.startswith("data:"):
            try:
                data = json.loads(decoded[5:].strip())
                if data.get("type") == "completion":
                    text = data.get("completion", "")
                    if text:
                        full_text += text
                        print(text, end="", flush=True)  # 실시간 스트리밍 출력
            except json.JSONDecodeError:
                pass

# print(full_text)  # 전체 응답을 마지막에 출력하려면 주석 해제
```

---

## 오류 해결

| 오류 | 원인 |
|---|---|
| `401 Unauthorized` | `sessionKey`가 만료됨 — 개발자 도구에서 다시 추출 필요 |
| `404 Not Found` | `lastActiveOrg` 또는 `chat_session` 값이 올바르지 않음 |
| 빈 응답 | `anthropic-device-id` 또는 `user-agent` 값이 잘못됨 |
| Network 탭에 `completion` 없음 | **Preserve Log** 활성화 후 메시지 재전송 |

> **참고:** `sessionKey` 쿠키는 만료 기한이 있습니다. 요청이 실패하기 시작하면 개발자 도구에서 새 세션 키를 다시 추출하세요.