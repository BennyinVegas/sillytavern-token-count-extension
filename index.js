import { saveSettingsDebounced } from '../../../../script.js';
import { extension_settings, getContext } from '../../../extensions.js';

const MODULE = 'token_tracker';
const OR_API = 'https://openrouter.ai/api/v1';
const MODELS_CACHE_KEY = 'token_tracker_or_models';
const MODELS_TTL = 6 * 60 * 60 * 1000; // 6 hours
const GEN_URL = '/api/backends/chat-completions/generate';

/* ------------------------------------------------------------------ */
/* Settings / storage                                                  */
/* ------------------------------------------------------------------ */

const defaults = { enabled: true, days: {}, recent: [] };
const session = emptyBucket();
let range = 'today';

function cfg() {
    if (!extension_settings[MODULE]) extension_settings[MODULE] = structuredClone(defaults);
    for (const k of Object.keys(defaults)) {
        if (extension_settings[MODULE][k] === undefined) extension_settings[MODULE][k] = structuredClone(defaults[k]);
    }
    return extension_settings[MODULE];
}

function emptyBucket() {
    return { req: 0, pt: 0, ct: 0, cost: 0, models: {} };
}

function addTo(b, r) {
    b.req++; b.pt += r.pt; b.ct += r.ct; b.cost += r.cost;
    const key = `${r.model}|${r.provider || ''}`;
    const m = (b.models[key] ??= { req: 0, pt: 0, ct: 0, cost: 0 });
    m.req++; m.pt += r.pt; m.ct += r.ct; m.cost += r.cost;
}

function mergeInto(target, src) {
    target.req += src.req; target.pt += src.pt; target.ct += src.ct; target.cost += src.cost;
    for (const [k, v] of Object.entries(src.models)) {
        const m = (target.models[k] ??= { req: 0, pt: 0, ct: 0, cost: 0 });
        m.req += v.req; m.pt += v.pt; m.ct += v.ct; m.cost += v.cost;
    }
}

const dayKey = (d = new Date()) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

function bucketFor(r) {
    const s = cfg();
    if (r === 'session') return session;
    if (r === 'today') return s.days[dayKey()] ?? emptyBucket();
    const all = emptyBucket();
    for (const day of Object.values(s.days)) mergeInto(all, day);
    return all;
}

/* ------------------------------------------------------------------ */
/* OpenRouter pricing                                                  */
/* ------------------------------------------------------------------ */

let modelsMap = null;          // id -> { prompt, completion }
const endpointCache = new Map(); // `${id}` -> endpoints[] | null

async function loadModels(force = false) {
    if (modelsMap && !force) return modelsMap;
    try {
        if (!force) {
            const cached = JSON.parse(localStorage.getItem(MODELS_CACHE_KEY) || 'null');
            if (cached && Date.now() - cached.t < MODELS_TTL) {
                modelsMap = cached.map;
                return modelsMap;
            }
        }
    } catch { /* ignore */ }

    try {
        const res = await fetchOriginal(`${OR_API}/models`);
        const json = await res.json();
        const map = {};
        for (const m of json.data ?? []) {
            map[m.id] = {
                name: m.name,
                prompt: parseFloat(m.pricing?.prompt) || 0,
                completion: parseFloat(m.pricing?.completion) || 0,
            };
        }
        modelsMap = map;
        endpointCache.clear();
        try { localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify({ t: Date.now(), map })); } catch { /* quota */ }
    } catch (e) {
        console.warn('[TokenTracker] could not load OpenRouter models', e);
        modelsMap ??= {};
    }
    return modelsMap;
}

async function loadEndpoints(modelId) {
    if (endpointCache.has(modelId)) return endpointCache.get(modelId);
    let eps = null;
    try {
        const res = await fetchOriginal(`${OR_API}/models/${modelId}/endpoints`);
        if (res.ok) eps = (await res.json()).data?.endpoints ?? null;
    } catch { /* ignore */ }
    endpointCache.set(modelId, eps);
    return eps;
}

/** Returns { prompt, completion } USD-per-token, preferring the exact provider's price. */
async function getPricing(modelId, provider) {
    if (!modelId) return null;
    const models = await loadModels();
    const baseId = modelId in models ? modelId : modelId.replace(/:[a-z-]+$/i, '');

    if (provider) {
        const eps = await loadEndpoints(baseId);
        const p = provider.toLowerCase();
        const ep = eps?.find(e => {
            const n = (e.provider_name || '').toLowerCase();
            return n === p || n.startsWith(p) || p.startsWith(n);
        });
        if (ep?.pricing) {
            return { prompt: parseFloat(ep.pricing.prompt) || 0, completion: parseFloat(ep.pricing.completion) || 0 };
        }
    }
    const m = models[baseId];
    return m ? { prompt: m.prompt, completion: m.completion } : null;
}

/* ------------------------------------------------------------------ */
/* Token counting helpers                                              */
/* ------------------------------------------------------------------ */

async function countTokens(text) {
    if (!text) return 0;
    try {
        const ctx = getContext();
        if (ctx.getTokenCountAsync) return await ctx.getTokenCountAsync(text);
        if (ctx.getTokenCount) return ctx.getTokenCount(text);
    } catch { /* fall through */ }
    return Math.ceil(text.length / 4);
}

function promptText(body) {
    const msgs = body?.messages;
    if (!Array.isArray(msgs)) return typeof body?.prompt === 'string' ? body.prompt : '';
    return msgs.map(m => {
        if (typeof m.content === 'string') return m.content;
        if (Array.isArray(m.content)) return m.content.map(p => p.text || '').join('\n');
        return '';
    }).join('\n');
}

/* ------------------------------------------------------------------ */
/* Network interception                                                */
/* ------------------------------------------------------------------ */

const fetchOriginal = window.fetch.bind(window);

window.fetch = async function (input, init) {
    const res = await fetchOriginal(input, init);
    try {
        const url = typeof input === 'string' ? input : (input?.url ?? String(input));
        if (cfg().enabled && url.includes(GEN_URL) && res.ok) {
            // Fire and forget; never block or break SillyTavern's own handling.
            inspectResponse(res.clone(), init).catch(e => console.warn('[TokenTracker]', e));
        }
    } catch (e) {
        console.warn('[TokenTracker]', e);
    }
    return res;
};

async function inspectResponse(res, init) {
    let reqBody = {};
    try { reqBody = JSON.parse(init?.body); } catch { /* not JSON */ }

    const info = {
        model: reqBody.model || 'unknown',
        source: reqBody.chat_completion_source || 'unknown',
        provider: null,
        usage: null,
        text: '',
    };

    const type = res.headers.get('content-type') || '';
    if (type.includes('text/event-stream')) {
        try {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();
                for (const line of lines) consumeSseLine(line, info);
            }
            consumeSseLine(buf, info);
        } catch { /* aborted stream: use what we have */ }
    } else {
        try {
            const j = await res.json();
            info.model = j.model || info.model;
            info.provider = j.provider || null;
            info.usage = j.usage || null;
            info.text = j.choices?.[0]?.message?.content || j.choices?.[0]?.text || '';
        } catch { return; }
    }

    await record(info, reqBody);
}

function consumeSseLine(line, info) {
    line = line.trim();
    if (!line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    if (!data || data === '[DONE]') return;
    try {
        const j = JSON.parse(data);
        if (j.model) info.model = j.model;
        if (j.provider) info.provider = j.provider;
        if (j.usage) info.usage = j.usage;
        const c = j.choices?.[0];
        const piece = c?.delta?.content ?? c?.text ?? '';
        if (typeof piece === 'string') info.text += piece;
    } catch { /* partial / non-JSON line */ }
}

async function record(info, reqBody) {
    const u = info.usage;
    let estimated = false;

    let pt = u?.prompt_tokens;
    let ct = u?.completion_tokens;
    if (pt == null) { pt = await countTokens(promptText(reqBody)); estimated = true; }
    if (ct == null) { ct = await countTokens(info.text); estimated = true; }

    let cost = typeof u?.cost === 'number' ? u.cost : null;
    let priced = cost !== null;

    if (cost === null && info.source === 'openrouter') {
        const p = await getPricing(info.model, info.provider);
        if (p) { cost = pt * p.prompt + ct * p.completion; priced = true; estimated = true; }
    }
    if (cost === null) cost = 0;

    const rec = { model: info.model, provider: info.provider, pt, ct, cost };
    const s = cfg();
    const key = dayKey();
    addTo((s.days[key] ??= emptyBucket()), rec);
    addTo(session, rec);

    s.recent.unshift({ t: Date.now(), ...rec, estimated, priced });
    s.recent.length = Math.min(s.recent.length, 25);

    saveSettingsDebounced();
    render();
}

/* ------------------------------------------------------------------ */
/* UI                                                                  */
/* ------------------------------------------------------------------ */

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtN = (n) => Math.round(n).toLocaleString();
const fmtCost = (n) => `$${n < 1 ? n.toFixed(4) : n.toFixed(2)}`;

function render() {
    const root = document.getElementById('tt_root');
    if (!root) return;

    const b = bucketFor(range);
    const rows = Object.entries(b.models)
        .sort((a, c) => c[1].cost - a[1].cost)
        .map(([key, m]) => {
            const [model, provider] = key.split('|');
            return `<tr>
                <td class="tt_model" title="${esc(model)}">${esc(model)}</td>
                <td>${esc(provider) || '<span class="tt_dim">—</span>'}</td>
                <td>${fmtN(m.req)}</td>
                <td>${fmtN(m.pt)}</td>
                <td>${fmtN(m.ct)}</td>
                <td>${fmtCost(m.cost)}</td>
            </tr>`;
        }).join('');

    root.querySelector('#tt_summary').innerHTML = `
        <div class="tt_card"><span>Requests</span><b>${fmtN(b.req)}</b></div>
        <div class="tt_card"><span>Input tokens</span><b>${fmtN(b.pt)}</b></div>
        <div class="tt_card"><span>Output tokens</span><b>${fmtN(b.ct)}</b></div>
        <div class="tt_card"><span>Cost</span><b>${fmtCost(b.cost)}</b></div>`;

    root.querySelector('#tt_models tbody').innerHTML =
        rows || '<tr><td colspan="6" class="tt_dim">No data yet.</td></tr>';

    const recent = cfg().recent.slice(0, 8).map(r => `<tr>
        <td>${new Date(r.t).toLocaleTimeString()}</td>
        <td class="tt_model" title="${esc(r.model)}">${esc(r.model)}</td>
        <td>${esc(r.provider) || '—'}</td>
        <td>${r.estimated && !r.priced ? '~' : ''}${fmtN(r.pt)} / ${fmtN(r.ct)}</td>
        <td>${r.priced ? (r.estimated ? '~' : '') + fmtCost(r.cost) : '<span class="tt_dim">n/a</span>'}</td>
    </tr>`).join('');
    root.querySelector('#tt_recent tbody').innerHTML =
        recent || '<tr><td colspan="5" class="tt_dim">No requests yet.</td></tr>';

    const today = bucketFor('today');
    const title = document.getElementById('tt_title_cost');
    if (title) title.textContent = `${fmtCost(today.cost)} today`;
}

function exportCsv() {
    const lines = ['date,model,provider,requests,prompt_tokens,completion_tokens,cost_usd'];
    for (const [day, b] of Object.entries(cfg().days).sort()) {
        for (const [key, m] of Object.entries(b.models)) {
            const [model, provider] = key.split('|');
            lines.push([day, model, provider, m.req, m.pt, m.ct, m.cost.toFixed(6)]
                .map(v => `"${String(v).replace(/"/g, '""')}"`).join(','));
        }
    }
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: 'token-usage.csv' });
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
}

function buildUi() {
    const html = `
    <div id="tt_root" class="inline-drawer">
        <div class="inline-drawer-toggle inline-drawer-header">
            <b>Token &amp; Cost Tracker <span id="tt_title_cost" class="tt_dim"></span></b>
            <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
        </div>
        <div class="inline-drawer-content">
            <label class="checkbox_label">
                <input type="checkbox" id="tt_enabled"><span>Enable tracking</span>
            </label>
            <div class="tt_toolbar">
                <select id="tt_range" class="text_pole">
                    <option value="session">This session</option>
                    <option value="today">Today</option>
                    <option value="all">All time</option>
                </select>
                <div id="tt_refresh" class="menu_button" title="Re-download OpenRouter pricing">Refresh prices</div>
                <div id="tt_export" class="menu_button">Export CSV</div>
                <div id="tt_reset" class="menu_button">Reset</div>
            </div>
            <div id="tt_summary"></div>
            <h4>By model / provider</h4>
            <div class="tt_scroll">
                <table id="tt_models" class="tt_table">
                    <thead><tr><th>Model</th><th>Provider</th><th>Req</th><th>In</th><th>Out</th><th>Cost</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
            <h4>Recent requests</h4>
            <div class="tt_scroll">
                <table id="tt_recent" class="tt_table">
                    <thead><tr><th>Time</th><th>Model</th><th>Provider</th><th>In / Out</th><th>Cost</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
            <small class="tt_dim">
                Cost comes from OpenRouter's reported usage when available; otherwise it is estimated (~) from
                the provider's listed price. Non-OpenRouter sources show tokens only.
            </small>
        </div>
    </div>`;

    const host = document.getElementById('extensions_settings2') || document.getElementById('extensions_settings');
    host.insertAdjacentHTML('beforeend', html);

    const $ = (id) => document.getElementById(id);
    $('tt_enabled').checked = cfg().enabled;
    $('tt_enabled').addEventListener('change', (e) => { cfg().enabled = e.target.checked; saveSettingsDebounced(); });
    $('tt_range').value = range;
    $('tt_range').addEventListener('change', (e) => { range = e.target.value; render(); });
    $('tt_refresh').addEventListener('click', async () => {
        await loadModels(true);
        toastr.success('OpenRouter pricing refreshed');
    });
    $('tt_export').addEventListener('click', exportCsv);
    $('tt_reset').addEventListener('click', () => {
        if (!confirm('Delete all tracked usage data?')) return;
        extension_settings[MODULE] = structuredClone(defaults);
        Object.assign(session, emptyBucket());
        saveSettingsDebounced();
        render();
    });
    render();
}

jQuery(() => {
    buildUi();
    loadModels(); // warm the pricing cache
    console.log('[TokenTracker] loaded');
});
