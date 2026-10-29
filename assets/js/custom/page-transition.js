// Turbo로 페이지를 이동하면 전체 새로고침이 일어나지 않으므로,
// 새 페이지마다 해줘야 하는 작업(수식 렌더링, 방문 기록)을 여기서 처리한다.
(function () {
    var firstLoad = true;

    document.addEventListener('turbo:load', function () {
        // 첫 로드는 MathJax와 gtag가 각자 알아서 처리한다.
        if (firstLoad) {
            firstLoad = false;
            return;
        }

        if (window.MathJax && window.MathJax.Hub) {
            window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
        }

        if (typeof window.gtag === 'function') {
            window.gtag('event', 'page_view', {
                page_location: location.href,
                page_path: location.pathname,
                page_title: document.title
            });
        }
    });
})();
