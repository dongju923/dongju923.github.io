// 사이트 BGM. 페이지 이동은 Turbo가 본문만 교체하므로 이 스크립트(head)는 한 번만 실행되고,
// Audio 객체가 DOM 밖에 있어서 페이지를 넘겨도 재생이 끊기지 않는다.
(function () {
    if (window.__bgm) return;

    var SRC = '/assets/audio/usagi-rap.mp3';
    var KEY_OFF = 'bgm-off';     // 사용자가 버튼으로 끈 경우 (localStorage)
    var KEY_TIME = 'bgm-time';   // 새로고침 시 이어듣기용 재생 위치 (sessionStorage)

    var audio = new Audio(SRC);
    audio.loop = true;
    audio.volume = 0.35;
    audio.preload = 'metadata';
    window.__bgm = audio;

    function safe(fn) {
        try { return fn(); } catch (e) { return null; }
    }
    function isOff() {
        return safe(function () { return localStorage.getItem(KEY_OFF) === '1'; }) === true;
    }
    function setOff(off) {
        safe(function () {
            if (off) localStorage.setItem(KEY_OFF, '1');
            else localStorage.removeItem(KEY_OFF);
        });
    }

    var saved = parseFloat(safe(function () { return sessionStorage.getItem(KEY_TIME); }));
    if (saved > 0) {
        audio.addEventListener('loadedmetadata', function () {
            if (saved < audio.duration) audio.currentTime = saved;
        }, { once: true });
    }
    window.addEventListener('pagehide', function () {
        safe(function () { sessionStorage.setItem(KEY_TIME, String(audio.currentTime)); });
    });

    function play() {
        var p = audio.play();
        if (p && p.catch) p.catch(function () {});  // 브라우저가 막으면 다음 상호작용에서 다시 시도
    }

    // 브라우저 정책상 사용자 상호작용이 있어야 소리를 낼 수 있으므로, 첫 클릭/터치/키 입력에 재생한다.
    function onInteract(e) {
        if (!audio.paused || isOff()) return;
        if (e.target && e.target.closest && e.target.closest('.bgm-toggle')) return;
        play();
    }
    ['pointerdown', 'pointerup', 'touchend', 'keydown', 'click'].forEach(function (type) {
        document.addEventListener(type, onInteract, true);
    });

    function render() {
        var btn = document.querySelector('.bgm-toggle');
        if (!btn) {
            btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'bgm-toggle';
            btn.innerHTML = '<i aria-hidden="true"></i>';
            btn.addEventListener('click', function () {
                if (audio.paused) {
                    setOff(false);
                    play();
                } else {
                    setOff(true);
                    audio.pause();
                }
            });
            document.body.appendChild(btn);
        }
        var playing = !audio.paused;
        btn.classList.toggle('is-playing', playing);
        btn.querySelector('i').className = playing ? 'fas fa-volume-up' : 'fas fa-volume-mute';
        btn.title = playing ? 'BGM 끄기' : 'BGM 켜기';
        btn.setAttribute('aria-label', btn.title);
    }

    audio.addEventListener('play', render);
    audio.addEventListener('pause', render);
    document.addEventListener('turbo:load', render);
    document.addEventListener('DOMContentLoaded', render);
})();
