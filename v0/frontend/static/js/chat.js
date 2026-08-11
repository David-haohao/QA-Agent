/* ============================================================
   智能问答系统 — SSE流式客户端 (Claude Code Web 风格)
   组件化渲染: Thinking Block → Tool Call → Streaming Answer
   支持: 历史记忆 · 中文思考 · 来源点击 · 用户反馈
   ============================================================ */

// ===== State =====
const STATE = {
    sessionId: '',
    isStreaming: false,
    currentMessageEl: null,
    currentQuery: '',       // 用户当前问题
    currentAnswer: '',      // 系统当前回答
    queryId: '',
    sources: [],            // 来源文档列表 [{name, path, url}, ...]
    followups: [],
    elapsedMs: 0,
};

// ===== Init =====
document.addEventListener('DOMContentLoaded', function () {
    STATE.sessionId = getCookie('session_id') || '';
    if (!STATE.sessionId) {
        fetch('/api/session', { method: 'POST', body: new FormData() })
            .then(r => r.json())
            .then(d => { STATE.sessionId = d.session_id; updateStatus(); });
    }
    updateStatus();
    loadSuggestions();
    setupInput();
    setupButtons();
});

function setupInput() {
    const ta = document.getElementById('query-input');
    ta.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    ta.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
    });
}

function setupButtons() {
    document.getElementById('btn-send').addEventListener('click', sendQuery);
    document.getElementById('btn-new-session').addEventListener('click', newSession);
    document.getElementById('btn-refresh').addEventListener('click', loadSuggestions);
}

function getCookie(n) {
    const v = '; ' + document.cookie;
    const p = v.split('; ' + n + '=');
    return p.length === 2 ? p.pop().split(';').shift() : '';
}

// ===== Send Query =====
async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query || STATE.isStreaming) return;

    STATE.isStreaming = true;
    STATE.currentQuery = query;    // 保存用户问题（供反馈提交时使用）
    STATE.currentAnswer = '';
    STATE.sources = [];
    STATE.followups = [];
    STATE._analyticsSent = false;  // 每个问题独立发送埋点
    input.value = '';
    input.disabled = true;
    document.getElementById('btn-send').disabled = true;
    document.getElementById('welcome-screen').classList.add('hidden');
    document.getElementById('status-dot').classList.add('streaming');

    // Append user message (avatar on right — CSS row-reverse)
    appendMessage('user', query);
    // Create assistant message container (avatar on left)
    const msgEl = createAssistantMessage();
    STATE.currentMessageEl = msgEl;
    STATE.blockIdx = 0;

    const formData = new FormData();
    formData.append('query', query);
    formData.append('session_id', STATE.sessionId);

    try {
        const response = await fetch('/api/chat', { method: 'POST', body: formData });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            let currentEvent = '';
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.substring(7);
                } else if (line.startsWith('data: ') && currentEvent) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        handleEvent(currentEvent, data, msgEl);
                    } catch (e) { /* skip parse errors */ }
                    currentEvent = '';
                }
            }
        }

        // 处理 stream 结束时 buffer 中可能残留的最后一个事件
        if (buffer) {
            const lines = buffer.split('\n');
            let currentEvent = '';
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.substring(7);
                } else if (line.startsWith('data: ') && currentEvent) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        handleEvent(currentEvent, data, msgEl);
                    } catch (e) { /* skip parse errors */ }
                    currentEvent = '';
                }
            }
        }
    } catch (error) {
        appendBlock(msgEl, 'error', error.message);
    } finally {
        finishStreaming(msgEl);
    }
}

// ===== SSE Event Router =====
function handleEvent(event, data, msgEl) {
    switch (event) {
        case 'message_start':
            STATE.queryId = data.message_id || '';
            STATE.elapsedMs = data.elapsed_ms || 0;
            break;

        case 'thinking':
            appendBlock(msgEl, 'thinking', data);
            break;

        case 'tool_use':
            appendBlock(msgEl, 'tool_use', data);
            break;

        case 'tool_result':
            // 更新匹配的 tool_use 块，填充结果内容
            appendBlock(msgEl, 'tool_result', data);
            break;

        case 'content_block_delta':
            if (data.text) {
                STATE.currentAnswer += data.text;
                appendOrUpdateTextBlock(msgEl);
            }
            break;

        case 'content_block_stop':
            finishTextBlock(msgEl);
            break;

        case 'sources':
            // Only store sources, render at finish time (no duplicates)
            STATE.sources = data.sources || [];
            break;

        case 'followup':
            STATE.followups = data.questions || [];
            break;

        case 'status':
            document.getElementById('status-message').textContent = data.message || '';
            document.getElementById('status-time').textContent = ((data.elapsed_ms || 0) / 1000).toFixed(1) + 's';
            break;

        case 'message_stop':
            STATE.elapsedMs = data.elapsed_ms || 0;
            break;

        case 'done':
            finishStreaming(msgEl);
            break;

        case 'error':
            appendBlock(msgEl, 'error', data.message || '未知错误');
            break;
    }
}

// ===== Block Renderers =====
function appendBlock(msgEl, type, data) {
    const body = msgEl.querySelector('.message-body');
    const div = document.createElement('div');
    div.className = 'block-wrapper';

    switch (type) {
        case 'thinking':
            div.innerHTML = renderThinkingBlock(data.text, data.elapsed_ms);
            div.querySelector('.thinking-header').addEventListener('click', function () {
                const c = div.querySelector('.thinking-content');
                const a = div.querySelector('.thinking-arrow');
                c.classList.toggle('collapsed');
                a.classList.toggle('collapsed');
            });
            break;

        case 'tool_use':
            div.innerHTML = renderToolBlock(data);
            div.querySelector('.tool-card-header').addEventListener('click', function () {
                const c = div.querySelector('.tool-card-body');
                c.classList.toggle('collapsed');
            });
            // 存储 tool_call_id 用于后续 tool_result 匹配
            if (data.tool_call_id) {
                div.dataset.toolCallId = data.tool_call_id;
            }
            break;

        case 'tool_result':
            // tool_result 更新匹配的 tool_use block 而非创建新块
            _updateToolResultBlock(body, data);
            scrollToBottom();
            return;  // 不添加到DOM（更新已有块）

        case 'error':
            div.innerHTML = `<div class="error-block">${escapeHtml(data)}</div>`;
            break;
    }

    // 确保新块始终插入在 .text-block 之前（如果有的话）
    // 这样思考过程和工具调用的块显示在答案光标之前
    const textBlock = body.querySelector('.text-block');
    if (textBlock) {
        body.insertBefore(div, textBlock);
    } else {
        body.appendChild(div);
    }
    scrollToBottom();
}

// 更新匹配的 tool_use block，填充工具返回结果
function _updateToolResultBlock(body, data) {
    const blocks = body.querySelectorAll('.block-wrapper');
    // 优先按 tool_call_id 精确匹配
    if (data.tool_call_id) {
        for (let i = blocks.length - 1; i >= 0; i--) {
            if (blocks[i].dataset.toolCallId === data.tool_call_id) {
                _applyToolResult(blocks[i], data);
                return;
            }
        }
    }
    // 兜底：从后向前查找第一个 running 状态的工具块
    for (let i = blocks.length - 1; i >= 0; i--) {
        const badge = blocks[i].querySelector('.tool-card-badge');
        if (badge && badge.classList.contains('running')) {
            _applyToolResult(blocks[i], data);
            return;
        }
    }
}

function _applyToolResult(block, data) {
    const badge = block.querySelector('.tool-card-badge');
    if (badge) {
        badge.classList.remove('running');
        badge.classList.add('completed');
        badge.textContent = '完成';
    }
    // 填充工具返回结果到卡片内容区
    const bodyEl = block.querySelector('.tool-card-body');
    if (bodyEl && data.result) {
        // 显示工具返回的摘要（前300字符）
        const resultText = data.result.length > 300
            ? data.result.substring(0, 300) + '...'
            : data.result;
        bodyEl.innerHTML = `<pre>${escapeHtml(resultText)}</pre>`;
        // 展开显示结果
        bodyEl.classList.remove('collapsed');
    }
}

function appendOrUpdateTextBlock(msgEl) {
    const body = msgEl.querySelector('.message-body');
    let textBlock = body.querySelector('.text-block');
    if (!textBlock) {
        textBlock = document.createElement('div');
        textBlock.className = 'text-block streaming-text cursor-blink';
        body.appendChild(textBlock);
    }
    // 确保 text-block 始终是 body 的最后一个子元素
    // 这样答案流式光标始终显示在 thinking/tool_result 块之后
    if (textBlock !== body.lastElementChild) {
        body.appendChild(textBlock);
    }
    textBlock.innerHTML = formatAnswer(STATE.currentAnswer);
    scrollToBottom();
}

function finishTextBlock(msgEl) {
    const body = msgEl.querySelector('.message-body');
    const tb = body.querySelector('.text-block');
    if (tb) tb.classList.remove('cursor-blink');
}

// ===== Finish =====
// 每个消息元素上的完成标记，解决多轮对话中finally+done双重调用和跨消息冲突
const _FINISHED_MESSAGES = new WeakSet();

function finishStreaming(msgEl) {
    // 用WeakSet做per-message去重，替代全局STATE.isStreaming guard
    // 确保每个消息的反馈栏只创建一次，且不会因done/finally调用顺序而出错
    if (!msgEl || _FINISHED_MESSAGES.has(msgEl)) return;
    _FINISHED_MESSAGES.add(msgEl);

    // 恢复UI状态（只在实际流式结束时执行一次）
    if (STATE.isStreaming) {
        STATE.isStreaming = false;
        document.getElementById('query-input').disabled = false;
        document.getElementById('btn-send').disabled = false;
        document.getElementById('status-dot').classList.remove('streaming');
        document.getElementById('status-message').textContent = '就绪';
        document.getElementById('status-time').textContent = (STATE.elapsedMs / 1000).toFixed(1) + 's';
        document.getElementById('query-input').focus();
    }

    try {
        finishTextBlock(msgEl);
        const body = msgEl.querySelector('.message-body');
        if (!body) return;

        // 从流式文本中剥离末尾的 **来源：** 块（由前端来源附加块替代）
        stripSourceSectionFromDOM(body);

        // Sources: 可点击超链接——在新标签页查看文档全文
        if (STATE.sources && STATE.sources.length > 0) {
            const srcDiv = document.createElement('div');
            srcDiv.className = 'sources-section';
            srcDiv.innerHTML = '<div class="sources-title">📎 参考来源（点击查看全文）</div>' +
                STATE.sources.map(s => {
                    const linkTarget = s.path || s.name;
                    const displayName = s.name;
                    return `<a class="source-link" href="/kb/view/${encodeURIComponent(linkTarget)}" target="_blank" rel="noopener">📄 ${escapeHtml(displayName)}</a>`;
                }).join('');
            body.appendChild(srcDiv);
        }

        // Followup 关联问题
        if (STATE.followups && STATE.followups.length > 0) {
            const fuDiv = document.createElement('div');
            fuDiv.className = 'followup-section';
            fuDiv.innerHTML = '<span class="followup-label">您可能还想问</span>' +
                STATE.followups.map(q =>
                    `<span class="followup-chip" data-q="${escapeHtml(q)}">${escapeHtml(q)}</span>`
                ).join('');
            fuDiv.querySelectorAll('.followup-chip').forEach(chip => {
                chip.addEventListener('click', function () {
                    document.getElementById('query-input').value = this.getAttribute('data-q');
                    sendQuery();
                });
            });
            body.appendChild(fuDiv);
        }

        // === 反馈按钮——每个回答必须有独立的反馈栏 ===
        // 使用闭包保存当前消息的快照数据，不受后续STATE变化影响
        const snapshot = {
            queryId: STATE.queryId,
            query: STATE.currentQuery,
            answer: STATE.currentAnswer,
            sources: JSON.stringify(STATE.sources || []),
        };

        const fbDiv = document.createElement('div');
        fbDiv.className = 'feedback-bar';
        fbDiv.dataset.queryId = snapshot.queryId;
        fbDiv.dataset.query = snapshot.query;
        fbDiv.dataset.answer = snapshot.answer;
        fbDiv.dataset.sources = snapshot.sources;
        fbDiv.innerHTML = `
            <span class="feedback-label">这个回答对您有帮助吗？</span>
            <button class="fb-btn fb-like">👍 有帮助</button>
            <button class="fb-btn fb-dislike">👎 没帮助</button>
            <button class="fb-btn fb-correct">📝 纠错反馈</button>
        `;
        body.appendChild(fbDiv);

        // 每个按钮绑定到自己的fbDiv（闭包捕获，不依赖DOM查询）
        fbDiv.querySelector('.fb-like').addEventListener('click', function() {
            submitFeedback(1, fbDiv);
        });
        fbDiv.querySelector('.fb-dislike').addEventListener('click', function() {
            submitFeedback(0, fbDiv);
        });
        fbDiv.querySelector('.fb-correct').addEventListener('click', function() {
            showCorrectionModal(fbDiv);
        });

        console.log('[QA] 反馈栏已创建 queryId=' + snapshot.queryId);
    } catch (e) {
        console.error('[QA] 创建反馈栏失败:', e);
    }

    // Analytics（只在流式状态恢复时发送一次）
    if (STATE.queryId && STATE.elapsedMs > 0 && !STATE._analyticsSent) {
        STATE._analyticsSent = true;
        postAnalytics('answer_displayed', STATE.queryId, {
            elapsed_ms: STATE.elapsedMs,
            source_count: (STATE.sources || []).length,
        });
    }
}

// ===== Issue4 修复: 点击来源链接显示详情 =====
async function showSourceDetail(filename) {
    try {
        const resp = await fetch('/api/source-detail?filename=' + encodeURIComponent(filename));
        const data = await resp.json();
        const msg = data.found
            ? `📄 ${escapeHtml(filename)}` +
              `\n\n知识库总文档数: ${data.kb_doc_count}` +
              `\n总文本块数: ${data.kb_chunk_count}` +
              `\n\n该文档已完整索引，可在知识库中检索到。`
            : `📄 ${escapeHtml(filename)}\n\n该文档暂未在知识库中找到，请确认知识库已构建。`;
        alert(msg);
    } catch (e) {
        alert('获取文档详情失败: ' + e.message);
    }
}

// ===== 反馈功能（使用fbDiv参数获取每个回答独立的数据，解决多轮对话DOM冲突） =====
async function submitFeedback(rating, fbDiv) {
    // 从当前反馈栏的dataset读取数据（每个回答独立绑定，不会被后续对话覆盖）
    const queryId = fbDiv.dataset.queryId;
    const query = fbDiv.dataset.query;
    const answer = fbDiv.dataset.answer;
    const sources = fbDiv.dataset.sources || '[]';

    const formData = new FormData();
    formData.append('query_id', queryId);
    formData.append('session_id', STATE.sessionId);
    formData.append('rating', rating);
    formData.append('query', query);
    formData.append('answer', answer);
    formData.append('source_docs', sources);

    try {
        await fetch('/api/feedback', { method: 'POST', body: formData });
        // 隐藏当前反馈栏的所有按钮，显示"已反馈，有帮助"
        const allBtns = fbDiv.querySelectorAll('.fb-btn');
        allBtns.forEach(b => b.style.display = 'none');
        const label = fbDiv.querySelector('.feedback-label');
        if (label) label.textContent = '✅ 已反馈，有帮助';
        postAnalytics(rating === 1 ? 'feedback_positive' : 'feedback_negative', queryId, { rating });
    } catch (e) {
        console.error('反馈提交失败', e);
    }
}

function showCorrectionModal(fbDiv) {
    const existing = document.getElementById('correction-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'correction-modal';
    modal.className = 'correction-overlay';
    modal.innerHTML = `
        <div class="correction-panel">
            <h3>📝 纠错与补充反馈</h3>
            <div class="correction-types">
                <label><input type="radio" name="corr_type" value="content_error" checked> 内容纠错</label>
                <label><input type="radio" name="corr_type" value="source_error"> 来源引用有误</label>
                <label><input type="radio" name="corr_type" value="content_supplement"> 内容补充</label>
                <label><input type="radio" name="corr_type" value="other"> 其他建议</label>
            </div>
            <textarea id="correction-text" placeholder="请描述您发现的问题或需要补充的内容..." rows="4"></textarea>
            <div class="correction-actions">
                <button class="fb-btn" id="btn-cancel-correct">取消</button>
                <button class="fb-btn fb-like" id="btn-submit-correct">提交反馈</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // 从当前反馈栏的dataset读取数据（每个回答独立绑定）
    const queryId = fbDiv.dataset.queryId;
    const query = fbDiv.dataset.query;
    const answer = fbDiv.dataset.answer;
    const sources = fbDiv.dataset.sources || '[]';

    modal.querySelector('#btn-cancel-correct').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });
    modal.querySelector('#btn-submit-correct').addEventListener('click', async () => {
        const corrType = modal.querySelector('input[name="corr_type"]:checked').value;
        const corrText = modal.querySelector('#correction-text').value.trim();
        if (!corrText) { alert('请填写反馈内容'); return; }

        const formData = new FormData();
        formData.append('query_id', queryId);
        formData.append('session_id', STATE.sessionId);
        formData.append('rating', -1);
        formData.append('comment', corrText);
        formData.append('correction_type', corrType);
        formData.append('correction_text', corrText);
        formData.append('query', query);
        formData.append('answer', answer);
        formData.append('source_docs', sources);

        try {
            await fetch('/api/feedback', { method: 'POST', body: formData });
            modal.remove();
            // 纠错反馈后统一显示"已反馈，有帮助"（使用传入的fbDiv）
            const allBtns = fbDiv.querySelectorAll('.fb-btn');
            allBtns.forEach(b => b.style.display = 'none');
            const label = fbDiv.querySelector('.feedback-label');
            if (label) {
                label.textContent = '✅ 已反馈，有帮助';
            } else {
                fbDiv.innerHTML = '<span class="feedback-label" style="color:var(--success)">✅ 已反馈，有帮助</span>';
            }
            postAnalytics('feedback_correction', queryId, { type: corrType });
        } catch (e) {
            alert('提交失败: ' + e.message);
        }
    });
}

// ===== HTML Templates =====
function renderThinkingBlock(text, elapsed) {
    const now = new Date();
    const time = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
    return `
<div class="thinking-block">
    <div class="thinking-header">
        <span class="thinking-icon">💭</span>
        <span class="thinking-label">思考过程</span>
        <span class="thinking-time">${time} · ${((elapsed||0)/1000).toFixed(1)}s</span>
        <span class="thinking-arrow">▼</span>
    </div>
    <div class="thinking-content">${escapeHtml(text||'')}</div>
</div>`;
}

function renderToolBlock(data) {
    const icons = {'search_knowledge_base':'🔍','multi_search_knowledge_base':'🔎','list_knowledge_base_sources':'📋'};
    const names = {'search_knowledge_base':'搜索知识库','multi_search_knowledge_base':'多角度搜索','list_knowledge_base_sources':'查看文档列表'};
    const icon = icons[data.tool_name] || '🔧';
    const name = names[data.tool_name] || data.tool_name || '工具调用';
    const badgeCls = data.status === 'running' ? 'running' : 'completed';
    const badgeTxt = data.status === 'running' ? '执行中' : '完成';
    return `
<div class="tool-card">
    <div class="tool-card-header">
        <span class="tool-card-icon">${icon}</span>
        <span class="tool-card-name">${name}</span>
        <span class="tool-card-badge ${badgeCls}">${badgeTxt}</span>
    </div>
    <div class="tool-card-body collapsed"><pre>${escapeHtml(data.tool_input||'')}</pre></div>
</div>`;
}

// ===== Message Creators =====
function appendMessage(role, content) {
    const list = document.getElementById('message-list');
    const now = new Date();
    const time = now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
    const row = document.createElement('div');
    row.className = 'message-row ' + (role === 'user' ? 'user' : 'assistant');
    row.innerHTML = `
        <div class="message-avatar">${role==='user'?'我':'AI'}</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="role-name">${role==='user'?'您':'智能助手'}</span>
                <span class="time-label">${time}</span>
            </div>
            <div class="message-content">${role==='user'?escapeHtml(content):''}</div>
        </div>
    `;
    list.appendChild(row);
    scrollToBottom();
    return row;
}

function createAssistantMessage() {
    const list = document.getElementById('message-list');
    const now = new Date();
    const time = now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="role-name">智能助手</span>
                <span class="time-label">${time}</span>
            </div>
        </div>
    `;
    list.appendChild(row);
    scrollToBottom();
    return row;
}

// ===== Text Formatting =====
function formatAnswer(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/\|(.+)\|/g, function(m){
        if (m.includes('---')) return '<hr>';
        const cells = m.split('|').filter(c=>c.trim());
        return '<tr>'+cells.map(c=>{const t=c.trim();return t.match(/^:?-{3,}:?$/)?'':`<td>${t}</td>`;}).join('')+'</tr>';
    });
    html = html.replace(/(<tr>.*?<\/tr>\s*){2,}/g, '<table>$&</table>');
    html = html.replace(/^---$/gm, '<hr>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>'+html+'</p>';
    html = html.replace(/<p><\/p>/g, '');
    return html;
}

// ===== Suggestions =====
async function loadSuggestions() {
    const btn = document.getElementById('btn-refresh');
    btn.classList.add('spinning');
    setTimeout(()=>btn.classList.remove('spinning'),600);
    const list = document.getElementById('suggestion-list');
    try {
        const resp = await fetch('/api/suggestions');
        const data = await resp.json();
        const qs = data.questions || [];
        if (qs.length===0) { list.innerHTML='<div class="suggestion-skeleton">暂无推荐</div>'; return; }
        list.innerHTML = qs.map((q,i)=>`<div class="suggestion-item" data-q="${escapeHtml(q)}">${i+1}. ${escapeHtml(q)}</div>`).join('');
        list.querySelectorAll('.suggestion-item').forEach(item=>{
            item.addEventListener('click',function(){
                document.getElementById('query-input').value=this.getAttribute('data-q');
                sendQuery();
            });
        });
    } catch(e) { list.innerHTML='<div class="suggestion-skeleton">加载失败</div>'; }
}

// ===== Session =====
function newSession() {
    fetch('/api/session',{method:'POST',body:new FormData()})
        .then(r=>r.json())
        .then(data=>{
            STATE.sessionId=data.session_id;
            document.getElementById('message-list').innerHTML='';
            document.getElementById('welcome-screen').classList.remove('hidden');
            updateStatus();
            loadSuggestions();
        });
}

function updateStatus() {
    document.getElementById('status-session').textContent='会话: '+(STATE.sessionId||'...').substring(0,12);
    document.getElementById('status-message').textContent='就绪';
    document.getElementById('status-time').textContent='0.0s';
}

// ===== Analytics =====
function postAnalytics(eventType,queryId,metadata) {
    const fd=new FormData();
    fd.append('event_type',eventType);
    fd.append('session_id',STATE.sessionId);
    fd.append('source_query_id',queryId||'');
    fd.append('metadata',JSON.stringify(metadata||{}));
    navigator.sendBeacon('/api/analytics/event',fd);
}

// ===== 来源块剥离（前端DOM处理）=====
function stripSourceSectionFromDOM(body) {
    // 从渲染后的DOM中移除 **来源：** 块（由前端来源附加块替代）
    // 处理流式文本中Agent输出的末尾来源引用
    const textBlock = body.querySelector('.text-block');
    if (!textBlock) return;

    let html = textBlock.innerHTML;

    // 移除 **来源：** 或 来源： 行及之后的所有内容
    // 匹配 <strong>**来源：**</strong> 或纯文本 **来源：** 格式
    html = html.replace(/<strong>\*\*来源[：:]\*\*<\/strong>[\s\S]*$/, '');
    html = html.replace(/\*\*来源[：:]\*\*[\s\S]*$/, '');
    html = html.replace(/来源[：:][\s\S]*$/, '');

    // 清理末尾的空分隔线
    html = html.replace(/\n*<hr>\s*$/, '');
    html = html.replace(/<p>---<\/p>\s*$/, '');
    html = html.replace(/---\s*$/, '');

    textBlock.innerHTML = html;
}

// ===== Helpers =====
function escapeHtml(text) {
    const div=document.createElement('div');
    div.textContent=text;
    return div.innerHTML;
}
function scrollToBottom() {
    const area=document.getElementById('chat-area');
    requestAnimationFrame(()=>{area.scrollTop=area.scrollHeight;});
}
