// ============================================================
// 前端埋点 — 用户行为数据采集
// 为进化优化器提供数据支撑
// ============================================================

// 记录分析埋点事件
async function recordAnalyticsEvent(eventType, sourceQueryId, metadata) {
    const formData = new FormData();
    formData.append('event_type', eventType);
    formData.append('session_id', getSessionId());
    formData.append('source_query_id', sourceQueryId || '');
    formData.append('metadata', JSON.stringify(metadata || {}));

    // 非阻塞发送
    try {
        await fetch('/api/analytics/event', {
            method: 'POST',
            body: formData,
        });
    } catch (error) {
        // 埋点失败不影响主流程
    }
}

// 页面加载时记录页面访问事件
document.addEventListener('DOMContentLoaded', function() {
    recordAnalyticsEvent('page_view', '', { page: 'chat' });
});

// 监听来源链接点击
document.addEventListener('click', function(event) {
    if (event.target.classList.contains('source-link')) {
        recordAnalyticsEvent('source_link_click', currentQueryId, {
            link_url: event.target.href,
        });
    }
});

// 在用户离开页面时记录会话时长
let pageLoadTime = Date.now();
window.addEventListener('beforeunload', function() {
    const duration = Math.round((Date.now() - pageLoadTime) / 1000);
    // 使用navigator.sendBeacon保证在页面卸载时也能发送
    const formData = new FormData();
    formData.append('event_type', 'page_leave');
    formData.append('session_id', getSessionId());
    formData.append('metadata', JSON.stringify({ duration_seconds: duration }));
    navigator.sendBeacon('/api/analytics/event', formData);
});
