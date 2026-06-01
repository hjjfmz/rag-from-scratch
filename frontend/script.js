let currentSessionId = null;
let currentSessionIsEmpty = false;
const SESSION_ID_STORAGE_KEY = 'rag_from_scratch_session_id';
const STARTUP_SYNC_SEEN_KEY = 'rag_from_scratch_startup_sync_seen';

function loadSavedSessionId() {
    try {
        return localStorage.getItem(SESSION_ID_STORAGE_KEY);
    } catch {
        return null;
    }
}

function saveSessionId(id) {
    try {
        if (!id) {
            localStorage.removeItem(SESSION_ID_STORAGE_KEY);
        } else {
            localStorage.setItem(SESSION_ID_STORAGE_KEY, id);
        }
    } catch {
        return;
    }
}

async function loadSessions() {
    const res = await fetch('/api/sessions');
    const sessions = await res.json();
    const list = document.getElementById('session-list');
    list.innerHTML = '';

    sessions.forEach(s => {
        const div = document.createElement('div');
        div.className = `session-item ${s.session_id === currentSessionId ? 'active' : ''}`;
        div.innerText = s.title || 'Chat';
        div.onclick = () => loadSession(s.session_id);
        list.appendChild(div);
    });

    return sessions;
}

async function loadSession(id) {
    currentSessionId = id;
    saveSessionId(id);
    const res = await fetch(`/api/sessions/${id}`);
    if (!res.ok) {
        throw new Error('Session not found');
    }

    const session = await res.json();
    currentSessionIsEmpty = (session.messages.length === 0);
    document.getElementById('current-session-title').innerText = session.title;

    const container = document.getElementById('chat-container');
    container.innerHTML = '';

    session.messages.forEach(m => {
        appendFinalMessage(m.role, m.content);
    });

    await loadSessions();
}

async function newSession() {
    if (currentSessionIsEmpty) return;
    const res = await fetch('/api/sessions', { method: 'POST' });
    const session = await res.json();
    await loadSession(session.session_id);
}

function toggleCitations(headerEl) {
    const body = headerEl.nextElementSibling;
    const icon = headerEl.querySelector('.toggle-icon');
    if (body.classList.contains('open')) {
        body.classList.remove('open');
        icon.innerText = '▼';
    } else {
        body.classList.add('open');
        icon.innerText = '▲';
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function normalizeSourcePath(source) {
    const full = String(source || 'unknown').replaceAll('\\', '/');
    const parts = full.split('/');
    const base = parts[parts.length - 1] || full;
    return { full, base };
}

function formatScore(score) {
    const num = Number(score);
    if (Number.isFinite(num)) return num.toFixed(3);
    return '-';
}

function createCitationsHtml(citations) {
    if (!citations || citations.length === 0) {
        return '';
    }

    const safeCitations = citations.map((c, idx) => {
        const { full, base } = normalizeSourcePath(c.source);
        return {
            id: Number.isFinite(Number(c.id)) ? Number(c.id) : (idx + 1),
            base: escapeHtml(base),
            full: escapeHtml(full),
            score: formatScore(c.score),
            snippet: escapeHtml((c.snippet || '').trim())
        };
    });

    return `
        <div class="citations-container">
            <div class="citations-header" onclick="toggleCitations(this)">
                <div class="citations-header-left">
                    <span class="citations-title">References</span>
                    <span class="citations-count">${safeCitations.length}</span>
                </div>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="citations-body">
                ${safeCitations.map(c => `
                    <div class="citation-item">
                        <div class="citation-main">
                            <div class="citation-topline">
                                <span class="citation-index">#${c.id}</span>
                                <span class="citation-source" title="${c.full}">${c.base}</span>
                            </div>
                            <div class="citation-snippet">${c.snippet}</div>
                        </div>
                        <div class="citation-side">
                            <div class="citation-side-label">score</div>
                            <div class="citation-badge">${c.score}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function createProgressHtml(progressSteps) {
    if (!progressSteps || progressSteps.length === 0) {
        return '';
    }

    return `
        <div class="stream-status">
            <div class="stream-status-title">🪄 实时执行过程</div>
            <div class="progress-list">
                ${progressSteps.map(step => `
                    <div class="progress-step ${step.state}">
                        <div class="progress-step-text">${step.message}</div>
                        <div class="progress-step-state">${step.state === 'active' ? '进行中' : '完成'}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function appendFinalMessage(role, text) {
    const container = document.getElementById('chat-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message msg-${role}`;

    if (role === 'user') {
        msgDiv.innerText = text;
    } else {
        msgDiv.innerHTML = `<div class="content-body">${marked.parse(text)}</div>`;
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return msgDiv;
}

function ensureStreamingMessage(role = 'bot') {
    const container = document.getElementById('chat-container');
    let msgDiv = document.getElementById('streaming-msg');
    if (!msgDiv) {
        msgDiv = document.createElement('div');
        msgDiv.id = 'streaming-msg';
        msgDiv.className = `message msg-${role}`;
        container.appendChild(msgDiv);
    }

    if (!msgDiv.querySelector('.stream-shell')) {
        msgDiv.innerHTML = `
            <div class="stream-shell">
                <div class="stream-progress-host"></div>
                <div class="stream-notice-host"></div>
                <div class="stream-citations-host"></div>
                <div class="content-body answer-streaming is-live"></div>
            </div>
        `;
    }

    return msgDiv;
}

function renderStreamingMessage({ text = '', notice = null, citations = [], progressSteps = [], isFinal = false }) {
    const msgDiv = ensureStreamingMessage('bot');

    const progressHost = msgDiv.querySelector('.stream-progress-host');
    const noticeHost = msgDiv.querySelector('.stream-notice-host');
    const citationsHost = msgDiv.querySelector('.stream-citations-host');
    const bodyEl = msgDiv.querySelector('.content-body');

    progressHost.innerHTML = createProgressHtml(progressSteps);
    noticeHost.innerHTML = notice
        ? `<div class="notice-banner"><span>⚠️</span><span>${notice}</span></div>`
        : '';
    citationsHost.innerHTML = createCitationsHtml(citations);

    bodyEl.classList.toggle('is-live', !isFinal);
    if (isFinal) {
        bodyEl.innerHTML = marked.parse(text || '');
    } else {
        bodyEl.textContent = text || '...';
    }

    const container = document.getElementById('chat-container');
    container.scrollTop = container.scrollHeight;
    return msgDiv;
}

function pushProgressStep(progressSteps, stage, message) {
    const existing = progressSteps.find(step => step.stage === stage);
    if (existing) {
        existing.message = message;
        existing.state = stage.endsWith('.done') ? 'done' : 'active';
        progressSteps.forEach(step => {
            if (step !== existing && step.state === 'active') {
                step.state = 'done';
            }
        });
        return;
    }

    progressSteps.forEach(step => {
        if (step.state === 'active') {
            step.state = 'done';
        }
    });

    progressSteps.push({
        stage,
        message,
        state: stage.endsWith('.done') ? 'done' : 'active'
    });
}

function finalizeProgress(progressSteps) {
    progressSteps.forEach(step => {
        step.state = 'done';
    });
}

async function sendMessage() {
    const input = document.getElementById('message-input');
    const text = input.value.trim();
    if (!text) return;

    const sendBtn = document.getElementById('send-btn');
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    appendFinalMessage('user', text);
    currentSessionIsEmpty = false;

    const req = {
        message: text,
        session_id: currentSessionId,
        search_mode: document.getElementById('search-mode').value
    };

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });

        if (!response.ok || !response.body) {
            throw new Error('Stream request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let botText = '';
        let currentNotice = null;
        let currentCitations = [];
        const progressSteps = [];

        pushProgressStep(progressSteps, 'request.sent', '🚀 请求已发送，等待服务端响应...');
        renderStreamingMessage({
            text: '',
            notice: currentNotice,
            citations: currentCitations,
            progressSteps,
            isFinal: false
        });

        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') {
                    break;
                }

                try {
                    const data = JSON.parse(dataStr);
                    if (data.session_id && !currentSessionId) {
                        currentSessionId = data.session_id;
                        saveSessionId(currentSessionId);
                    }

                    if (data.type === 'progress') {
                        pushProgressStep(progressSteps, data.stage || 'progress', data.message || '处理中...');
                        renderStreamingMessage({
                            text: botText,
                            notice: currentNotice,
                            citations: currentCitations,
                            progressSteps,
                            isFinal: false
                        });
                    } else if (data.type === 'notice') {
                        currentNotice = data.message;
                        renderStreamingMessage({
                            text: botText,
                            notice: currentNotice,
                            citations: currentCitations,
                            progressSteps,
                            isFinal: false
                        });
                    } else if (data.type === 'citations') {
                        currentCitations = data.citations || [];
                        renderStreamingMessage({
                            text: botText,
                            notice: currentNotice,
                            citations: currentCitations,
                            progressSteps,
                            isFinal: false
                        });
                    } else if (data.type === 'answer.delta') {
                        botText += data.chunk || '';
                        renderStreamingMessage({
                            text: botText,
                            notice: currentNotice,
                            citations: currentCitations,
                            progressSteps,
                            isFinal: false
                        });
                    } else if (data.type === 'answer.final') {
                        botText = data.answer || botText;
                        finalizeProgress(progressSteps);
                        renderStreamingMessage({
                            text: botText,
                            notice: currentNotice,
                            citations: currentCitations,
                            progressSteps,
                            isFinal: true
                        });
                    }
                } catch (e) {
                    console.error('Parse error', dataStr, e);
                }
            }
        }

        const finalDiv = document.getElementById('streaming-msg');
        if (finalDiv) {
            finalDiv.removeAttribute('id');
        }

        await loadSessions();
    } catch (e) {
        console.error(e);
        appendFinalMessage('bot', 'Error communicating with server.');
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}

async function initApp() {
    const sessions = await loadSessions();
    const savedId = loadSavedSessionId();

    if (savedId) {
        try {
            await loadSession(savedId);
            await maybeShowStartupSync();
            return;
        } catch {
            saveSessionId(null);
        }
    }

    if (sessions && sessions.length > 0) {
        try {
            await loadSession(sessions[0].session_id);
            await maybeShowStartupSync();
            return;
        } catch {
            return;
        }
    }

    await newSession();
    await maybeShowStartupSync();
}

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function loadStartupSyncSeen() {
    try {
        return localStorage.getItem(STARTUP_SYNC_SEEN_KEY);
    } catch {
        return null;
    }
}

function saveStartupSyncSeen(value) {
    try {
        if (!value) {
            localStorage.removeItem(STARTUP_SYNC_SEEN_KEY);
        } else {
            localStorage.setItem(STARTUP_SYNC_SEEN_KEY, value);
        }
    } catch {
        return;
    }
}

function prependSystemCard(innerHtml) {
    const container = document.getElementById('chat-container');
    const card = document.createElement('div');
    card.className = 'message msg-bot';
    card.innerHTML = innerHtml;
    container.prepend(card);
}

async function maybeShowStartupSync() {
    let data;
    try {
        const res = await fetch('/api/startup/sync');
        if (!res.ok) return;
        data = await res.json();
    } catch {
        return;
    }

    if (!data || !data.changed || data.status !== 'reindexed') return;
    if (!Number(data.files_reindexed || 0)) return;

    const stamp = String(data.finished_at || data.started_at || '');
    const seen = loadStartupSyncSeen();
    if (seen && seen === stamp) return;

    const filesTotal = Number(data.files_total || 0);
    const filesReindexed = Number(data.files_reindexed || 0);
    const sampleDir = escapeHtml(data.sample_dir || 'sample_data');
    const fileList = Array.isArray(data.files) ? data.files : [];

    const steps = [
        { stage: 'scan', message: `🧾 扫描 ${sampleDir}：发现 ${filesTotal} 个文件`, state: 'done' },
        { stage: 'detect', message: `🔁 检测到未摄取或变更，已触发重新摄取`, state: 'done' },
        { stage: 'done', message: `✅ 已重新摄取 ${filesReindexed} 个文件（索引已更新）`, state: 'done' }
    ];

    const filesHtml = fileList.length
        ? `<details style="margin-top:10px;">
              <summary style="cursor:pointer; color:#cfcfcf; font-size:12px;">查看文件列表</summary>
              <div style="margin-top:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:12px; color:#b6b6b6; line-height:1.5;">
                  ${fileList.slice(0, 200).map(f => `<div>${escapeHtml(String(f))}</div>`).join('')}
              </div>
           </details>`
        : '';

    const html = `
        <div class="stream-status">
            <div class="stream-status-title">📥 启动时自动摄取（sample_data）</div>
            <div class="progress-list">
                ${steps.map(step => `
                    <div class="progress-step ${step.state}">
                        <div class="progress-step-text">${step.message}</div>
                        <div class="progress-step-state">完成</div>
                    </div>
                `).join('')}
            </div>
            ${filesHtml}
        </div>
    `;

    prependSystemCard(html);
    saveStartupSyncSeen(stamp || 'shown');
}

function showUpload() {
    document.getElementById('modal-overlay').style.display = 'block';
    document.getElementById('upload-modal').style.display = 'block';
    clearUploadError();
    selectUploadTab('file');
}

function hideUpload() {
    document.getElementById('modal-overlay').style.display = 'none';
    document.getElementById('upload-modal').style.display = 'none';
}

let selectedFiles = [];

function selectUploadTab(tab) {
    const fileTab = document.getElementById('tab-file');
    const urlTab = document.getElementById('tab-url');
    const filePane = document.getElementById('pane-file');
    const urlPane = document.getElementById('pane-url');

    clearUploadError();

    if (tab === 'url') {
        fileTab.classList.remove('active');
        urlTab.classList.add('active');
        filePane.classList.remove('active');
        urlPane.classList.add('active');
    } else {
        urlTab.classList.remove('active');
        fileTab.classList.add('active');
        urlPane.classList.remove('active');
        filePane.classList.add('active');
    }
}

function setUploadError(message) {
    const el = document.getElementById('upload-error');
    el.textContent = message || '';
    el.classList.toggle('show', Boolean(message));
}

function clearUploadError() {
    setUploadError('');
}

function isAllowedFile(file) {
    const name = String(file.name || '').toLowerCase();
    return name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.pdf');
}

function renderSelectedFiles() {
    const host = document.getElementById('selected-files');
    host.innerHTML = '';

    selectedFiles.forEach((f, idx) => {
        const chip = document.createElement('div');
        chip.className = 'file-chip';
        chip.innerHTML = `<span>${escapeHtml(f.name)}</span><span class="x" title="Remove">×</span>`;
        chip.querySelector('.x').onclick = () => {
            selectedFiles.splice(idx, 1);
            renderSelectedFiles();
            updateUploadButtons();
        };
        host.appendChild(chip);
    });
}

function updateUploadButtons() {
    const btn = document.getElementById('upload-files-btn');
    btn.disabled = selectedFiles.length === 0;
}

function addFiles(files) {
    const incoming = Array.from(files || []);
    const bad = incoming.filter(f => !isAllowedFile(f));
    if (bad.length) {
        setUploadError('Only .txt / .md / .pdf files are supported.');
        return;
    }

    const existingNames = new Set(selectedFiles.map(f => f.name));
    incoming.forEach(f => {
        if (!existingNames.has(f.name)) {
            selectedFiles.push(f);
            existingNames.add(f.name);
        }
    });

    renderSelectedFiles();
    updateUploadButtons();
    clearUploadError();
}

async function uploadSelectedFiles() {
    if (!selectedFiles.length) return;
    clearUploadError();

    const btn = document.getElementById('upload-files-btn');
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    try {
        for (const file of selectedFiles) {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/documents/upload', { method: 'POST', body: formData });
            if (!res.ok) {
                const text = await res.text().catch(() => '');
                throw new Error(text || 'Upload failed');
            }
        }

        selectedFiles = [];
        renderSelectedFiles();
        updateUploadButtons();
        await loadSessions();
        hideUpload();
    } catch (e) {
        setUploadError(String(e?.message || e || 'Upload failed'));
    } finally {
        btn.textContent = 'Upload';
        btn.disabled = selectedFiles.length === 0;
    }
}

async function ingestUrl() {
    clearUploadError();
    const input = document.getElementById('url-input');
    const url = (input.value || '').trim();
    if (!url) {
        setUploadError('Please enter a URL.');
        return;
    }

    const btn = document.getElementById('upload-url-btn');
    btn.disabled = true;
    btn.textContent = 'Ingesting...';

    try {
        const res = await fetch('/api/documents/ingest-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(text || 'Ingest URL failed');
        }

        input.value = '';
        await loadSessions();
        hideUpload();
    } catch (e) {
        setUploadError(String(e?.message || e || 'Ingest URL failed'));
    } finally {
        btn.disabled = false;
        btn.textContent = 'Ingest URL';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            addFiles(e.target.files);
            e.target.value = '';
        });
    }

    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            addFiles(e.dataTransfer.files);
        });
    }

    const tabFile = document.getElementById('tab-file');
    const tabUrl = document.getElementById('tab-url');
    if (tabFile) tabFile.addEventListener('click', (e) => { e.preventDefault(); selectUploadTab('file'); });
    if (tabUrl) tabUrl.addEventListener('click', (e) => { e.preventDefault(); selectUploadTab('url'); });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const overlay = document.getElementById('modal-overlay');
            if (overlay && overlay.style.display === 'block') hideUpload();
        }
    });
});
