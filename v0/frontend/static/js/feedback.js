// ============================================================
// 反馈交互逻辑 — 用户反馈提交
// ============================================================

// 提交简单反馈(有帮助/没帮助)
async function submitFeedback(rating) {
    const formData = new FormData();
    formData.append('query_id', currentQueryId);
    formData.append('rating', rating);

    try {
        const response = await fetch('/api/feedback', { method: 'POST', body: formData });
        const result = await response.json();

        // 视觉反馈
        const buttons = document.querySelectorAll('.feedback-btn');
        if (rating === 1) {
            buttons[0].classList.add('active');
            buttons[0].textContent = 'N 感谢您的反馈！';
        } else {
            buttons[1].classList.add('active');
            buttons[1].textContent = 'N 感谢您的反馈！';
            showDislikeOptions();
        }

        // 埋点
        recordAnalyticsEvent(
            rating === 1 ? 'feedback_positive' : 'feedback_negative',
            currentQueryId,
            { rating: rating }
        );
    } catch (error) {
        console.error('提交反馈失败:', error);
    }
}

// 显示不满意原因选项
function showDislikeOptions() {
    const container = document.querySelector('.feedback-bar');
    const existing = document.getElementById('dislike-options');
    if (existing) existing.remove();

    const div = document.createElement('div');
    div.id = 'dislike-options';
    div.style.marginTop = '8px';
    div.innerHTML = `
        <p style="font-size:13px;margin-bottom:6px;">请问具体是哪里不满意？(可选)</p>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
            <label><input type="checkbox" value="答案不准确"> 答案不准确</label>
            <label><input type="checkbox" value="答案不完整"> 答案不完整</label>
            <label><input type="checkbox" value="未直接回答问题"> 未直接回答问题</label>
            <label><input type="checkbox" value="来源引用有误"> 来源引用有误</label>
        </div>
        <button class="feedback-btn" onclick="submitDislikeDetail()" style="margin-top:4px;">提交</button>
    `;
    container.appendChild(div);
}

// 提交差评详细原因
async function submitDislikeDetail() {
    const checkboxes = document.querySelectorAll('#dislike-options input:checked');
    const reasons = Array.from(checkboxes).map(c => c.value);

    const formData = new FormData();
    formData.append('query_id', currentQueryId);
    formData.append('rating', 0);
    formData.append('comment', '不满意原因: ' + reasons.join(', '));

    try {
        await fetch('/api/feedback', { method: 'POST', body: formData });
        document.getElementById('dislike-options').innerHTML = '<p style="font-size:13px;color:var(--success);">感谢您的反馈，我们会努力改进！</p>';
    } catch (error) {
        console.error('提交反馈失败:', error);
    }
}

// 显示纠错面板
function showCorrectionPanel() {
    const modal = document.createElement('div');
    modal.className = 'correction-modal active';
    modal.innerHTML = `
        <div class="correction-panel">
            <h3>N 纠错与补充反馈</h3>
            <div style="margin-bottom:8px;">
                <label><input type="radio" name="correction_type" value="content_error" checked> 内容纠错</label>
                <label><input type="radio" name="correction_type" value="source_error"> 来源纠错</label>
                <label><input type="radio" name="correction_type" value="content_supplement"> 内容补充</label>
                <label><input type="radio" name="correction_type" value="other"> 其他建议</label>
            </div>
            <textarea id="correction-text" placeholder="请描述您发现的问题或需要补充的内容"></textarea>
            <div class="correction-actions">
                <button class="feedback-btn" onclick="this.closest('.correction-modal').remove()">取消</button>
                <button class="feedback-btn" style="background:var(--primary);color:white;" onclick="submitCorrection()">提交反馈</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    modal.addEventListener('click', function(e) {
        if (e.target === modal) modal.remove();
    });
}

// 提交纠错反馈
async function submitCorrection() {
    const correctionType = document.querySelector('input[name="correction_type"]:checked')?.value || '';
    const correctionText = document.getElementById('correction-text').value.trim();

    const formData = new FormData();
    formData.append('query_id', currentQueryId);
    formData.append('rating', -1);
    formData.append('comment', `[${correctionType}] ${correctionText}`);

    try {
        await fetch('/api/feedback', { method: 'POST', body: formData });
        document.querySelector('.correction-modal').remove();
        // 提示
        const feedbackBar = document.querySelector('.feedback-bar');
        const hint = document.createElement('span');
        hint.style.fontSize = '13px';
        hint.style.color = 'var(--success)';
        hint.textContent = '感谢您的纠错反馈！';
        feedbackBar.appendChild(hint);
    } catch (error) {
        console.error('提交纠错失败:', error);
    }
}
