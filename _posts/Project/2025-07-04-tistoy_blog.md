---
title: "requests 로 tistory 블로그 자동으로 포스팅 하기"
categories: Project
toc: true
toc_sticky: true
---

요즘 인스타그램에 블로그 자동화로 월 몇백씩 번다는 광고가 수도 없이 많이 보여서 직접 돈주고 강의 듣기는 싫고 해서 대충 심심풀이로 자동화 포스팅 하는 
코드를 만들었다!!

### requests 란?
Python에서 HTTP 요청을 간편하게 보낼 수 있게 해주는 라이브러리 이다. 웹 서버와 통신할 때 가장 많이 사용되는 도구이다.  
간단한 문법으로 HTTP요청을 할 수 있고 `GET`, `POST` 등의 메서드를 지원한다.  
`POST`는 서버에 요청할 거, `GET` 은 서버에서 받아올거 라고 생각하면 편하다.  


### tistory에서의 requests 동작
tistory 블로그 작성페이지에서 `F12`를 눌러서 `개발자도구`를 들어간 다음 위쪽에 `Network` 를 클릭하면 현재 페이지에서 일어나는 모든 HTTP 요청을 확인 할 수 있다.  
이때 `Preserve log`에 체크를 해서 현재 시점의 요청 말고 이전시점의 요청도 확인할 수 있게 해준다.  
아무 글이나 작성하고 포스팅을 누르게 되면 `Network` 탭에 `post.json` 이라는 요청이 뜨는걸 볼 수 있다.  
이 POST 요청은 이제 우리가 글을 쓰고 포스팅을 누르게 되면 tistory서버에 post요청을 보내서 내가 쓴글좀 게시해줘 라고 말하는 것과 같다.  
이제 이걸 참고해서 코드를 작성하면 된다!  
![img.png](/assets/images/tis_blog/img2.png)  


### header와 payload
POST 요청을 보낼때는 header와 보낼 데이터가 필요하다. header는 누가 요청을 보냈는지, 요청의 형식이 무엇인지에 대한 일종의 메타데이터이다.  
payload는 요청의 실제 데이터이다. 즉, 블로그를 작성하는데에 있어서 제목, 내용, 태그 등등의 정보가 담겨있다.  
코드로 작성할 때, `post.json`에 있는 `Headers`에 있는 내용과 `Payload`에 있는 데이터를 참고하면 된다.  

### Header 작성
우선 헤더는 아래처럼 작성한다.  
```python
authority = "yourblog"   # post.json의 Request Headers에서 맨 위에 있는 authority임. 블로그이름.tistory.com 이다.
headers = {
    "Host": authority,
    "Cookie": cookie_str,
    "Sec-Ch-Ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "Accept": "application/json, text/plain, */*",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Accept-Language": "ko-KR",
    "Sec-Ch-Ua-Mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://" + authority,
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://" + authority + "/manage/newpost/?",
    "Accept-Encoding": "gzip, deflate, br",
    "Priority": "u=1, i"
}
```
여기서 Cookie 가 있는데, 이건 `post.json`에 있는 쿠키 값을 넣거나 `Application`창에서 쿠키를 찾아서 직접 입력해야하는데, 시간이 지남에 따라 쿠키는 바뀌기 때문에
일일히 노다가로 하기에는 너무 번거롭다.  
따라서 Selenium으로 자동화 하는 방법은 아래에서 설명하겠다.  

### Payload 작성
```python
post_data = {
            "id": "0",
            "title": '안녕하세요',
            "content": '테스트입니다.',
            "slogan": "test",
            "visibility": 20,
            "category": None,
            "tag": '태그',
            "published": 1, # 공개
            "password": "4zNTAyMz",  # 비공개 게시글이라면
            "uselessMarginForEntry": 1,
            "daumLike": None,
            "cclCommercial": 0,
            "cclDerive": 0,
            "thumbnail": None,
            "type": "post",
            "attachments": [],
            "recaptchaValue": "",
            "draftSequence": None,
            "challengeCode": ""
}
```
여기서 title은 제목, content는 내용, category는 카테고리, tag는 태그이다. published를 1로 하면 공개 포스팅이 된다.  
> 카테고리는 id가 필요한데 따로 어디서 보는 방법은 모르겠고 글쓰기 페이지에서 카테코리 고르는 곳에 개발자 도구로 코드를 확인해보면 각각의 카테고리에 대한 id가 있다.  
### 요청 보내기
```python
post_url = "https://" + authority + "/manage/post.json"
response = requests.post(post_url, headers=headers, json=post_data)
print(response.status_code)
```
요청은 이제 `requests.post()`로 보내고, url과 headers, data를 넣고 보내면 된다.  
`response.status_code` 가 200이라면 잘 전달된것이다!

### 쿠키 자동추출
```python
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
# 디버깅용 사용자 데이터 폴더
user_data_dir = r'C:\ChromeDebugTemp'

# 디버깅 포트 실행 명령어
chrome_process = subprocess.Popen([
    chrome_path,
    f'--remote-debugging-port=9222',
    f'--user-data-dir={user_data_dir}'
])

# Chrome 실행 대기 (2~3초 정도)
time.sleep(3)

# Selenium 연결
options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=options)
main_url = 'https://allnewsfast.tistory.com/manage/newpost'
driver.get(main_url)
wait = WebDriverWait(driver, 10)

# 로그인버튼 클릭
login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="cMain"]/div/div/div/div/a[2]')))
login_btn.click()

# 아이디 입력창 작성
id_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="loginId--1"]')))
id_field.send_keys('아이디 입력')

# 비밀번호 입력 창 작성
pass_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="password--2"]')))
pass_field.send_keys('패스워드 입력')

# 로그인
submit_btn = wait.until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="mainContent"]/div/div/form/div[4]/button[1]')))
submit_btn.click()

try:
    time.sleep(2)
    alert = driver.switch_to.alert
    alert.dismiss()
    print("Alert 삭제")
except NoAlertPresentException:
    pass

# 쿠키추출
cookies = driver.get_cookies()
cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
driver.quit()
```
selenium을 사용해서 tistory에 로그인을 하고 get_cookies()를 사용해서 현재 쿠키를 추출한다.  
일반 webdriver를 사용할 경우 계속 새롭게 로그인을 하기 때문에 로그인 인증이 계속 필요해서 디버그 크롬으로 하였다. 디버그 크롬은 그냥 우리가 흔히 사용하는
크롬(자동로그인 및 쿠키 저장 등등)이라고 생각하면 된다. 그래도 처음 한번 인증은 필요한듯 하다...

### 결과
![img.png](/assets/images/tis_blog/img3.png) 
성공적으로 포스팅 된것을 알 수 있다!


### 최종코드
```python
import time
import subprocess
import requests

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoAlertPresentException

authority = 'blogname.tistory.com'

chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
# 디버깅용 사용자 데이터 폴더
user_data_dir = r'C:\ChromeDebugTemp'

# 디버깅 포트 실행 명령어
chrome_process = subprocess.Popen([
    chrome_path,
    f'--remote-debugging-port=9222',
    f'--user-data-dir={user_data_dir}'
])

# Chrome 실행 대기 (2~3초 정도)
time.sleep(3)

# Selenium 연결
options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=options)
main_url = 'https://allnewsfast.tistory.com/manage/newpost'
driver.get(main_url)
wait = WebDriverWait(driver, 10)

# 로그인버튼 클릭
login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="cMain"]/div/div/div/div/a[2]')))
login_btn.click()

# 아이디 입력창 작성
id_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="loginId--1"]')))
id_field.send_keys('아이디 입력')

# 비밀번호 입력 창 작성
pass_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="password--2"]')))
pass_field.send_keys('패스워드 입력')

# 로그인
submit_btn = wait.until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="mainContent"]/div/div/form/div[4]/button[1]')))
submit_btn.click()

try:
    time.sleep(2)
    alert = driver.switch_to.alert
    alert.dismiss()
    print("Alert 삭제")
except NoAlertPresentException:
    pass

# 쿠키추출

cookies = driver.get_cookies()
cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
driver.quit()


headers = {
    "Host": authority,
    "Cookie": cookie_str,
    "Sec-Ch-Ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "Accept": "application/json, text/plain, */*",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Accept-Language": "ko-KR",
    "Sec-Ch-Ua-Mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://" + authority,
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://" + authority + "/manage/newpost/?",
    "Accept-Encoding": "gzip, deflate, br",
    "Priority": "u=1, i"
}

post_data = {
            "id": "0",
            "title": '안녕하세요',
            "content": '테스트입니다.',
            "slogan": "test",
            "visibility": 20,
            "category": None,
            "tag": '태그',
            "published": 1, # 공개
            "password": "4zNTAyMz",  # 비공개 게시글이라면
            "uselessMarginForEntry": 1,
            "daumLike": None,
            "cclCommercial": 0,
            "cclDerive": 0,
            "thumbnail": None,
            "type": "post",
            "attachments": [],
            "recaptchaValue": "",
            "draftSequence": None,
            "challengeCode": ""
}

post_url = "https://" + authority + "/manage/post.json"
response = requests.post(post_url, headers=headers, json=post_data)
print(response.text)
```
위 코드에서 authority와 아이디 비밀번호만 수정해서 넣자!!

### 한줄평
블로그로 수익창출은 너무 레드오션같다... 광고도 해야하고 블로그 홍보도 해야할듯...
이런걸로 돈 버는 사람은 극히 일부거나 인스타로 광고하는 사람들이 돈 벌듯....