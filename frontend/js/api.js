const API = {
  BASE: '/api',

  async _fetch(url, options = {}) {
    const res = await fetch(this.BASE + url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async uploadImage(file, name) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${this.BASE}/projects?name=${encodeURIComponent(name)}`, {
      method: 'POST', body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  getProject(id) { return this._fetch(`/projects/${id}`); },
  listProjects() { return this._fetch('/projects'); },
  getRegions(pid) { return this._fetch(`/projects/${pid}/regions`); },

  updateRegion(pid, rid, data) {
    return this._fetch(`/projects/${pid}/regions/${rid}`, {
      method: 'PATCH', body: JSON.stringify(data),
    });
  },

  batchUpdateRegions(pid, regionIds, style, status) {
    return this._fetch(`/projects/${pid}/batch-update-regions`, {
      method: 'POST', body: JSON.stringify({ region_ids: regionIds, style, status }),
    });
  },

  detect(pid, options = {}) {
    return this._fetch(`/projects/${pid}/detect`, {
      method: 'POST', body: JSON.stringify(options),
    });
  },

  ocr(pid) {
    return this._fetch(`/projects/${pid}/ocr`, { method: 'POST' });
  },

  inpaint(pid, simpleMode = false, regionIds = null) {
    const body = { method: simpleMode ? 'simple' : 'auto' };
    if (regionIds) body.region_ids = regionIds;
    return this._fetch(`/projects/${pid}/inpaint`, {
      method: 'POST', body: JSON.stringify(body),
    });
  },

  render(pid) {
    return this._fetch(`/projects/${pid}/render`, { method: 'POST' });
  },

  exportProject(pid, format = 'png', quality = 95) {
    return this._fetch(`/projects/${pid}/export`, {
      method: 'POST', body: JSON.stringify({ format, quality }),
    });
  },

  restoreRegion(pid, x, y, width, height) {
    return this._fetch(`/projects/${pid}/restore`, {
      method: 'POST', body: JSON.stringify({ x, y, width, height }),
    });
  },

  detectRegion(pid, x, y, width, height, mode = 'auto') {
    return this._fetch(`/projects/${pid}/detect-region`, {
      method: 'POST', body: JSON.stringify({ x, y, width, height, mode }),
    });
  },

  getImageUrl(pid, type) {
    return `${this.BASE}/projects/${pid}/image/${type}`;
  },

  getFonts() { return this._fetch('/fonts'); },
  correct(pid, options = {}) {
    return this._fetch(`/projects/${pid}/correct`, {
      method: 'POST', body: JSON.stringify(options),
    });
  },

  correctStream(pid, options = {}) {
    const controller = new AbortController();
    // Build clean body — strip callbacks before JSON serialization
    const body = JSON.stringify({
      provider: options.provider || 'deepseek',
      model: options.model || 'deepseek-v4-flash',
      auto_accept: options.auto_accept || false,
    });
    const promise = new Promise((resolve, reject) => {
      fetch(`${this.BASE}/projects/${pid}/correct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body,
        signal: controller.signal,
      }).then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'start' && options.onStart) {
                options.onStart(data.total);
              } else if (data.type === 'progress' && options.onProgress) {
                options.onProgress(data.completed, data.total);
              } else if (data.type === 'complete') {
                result = data.regions || [];
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Unknown error');
              }
            } catch (e) {
              if (e.message && !e.message.startsWith('Unexpected')) reject(e);
            }
          }
        }
        // Process any remaining buffer after stream ends
        if (buffer.startsWith('data: ')) {
          try {
            const data = JSON.parse(buffer.slice(6));
            if (data.type === 'complete') {
              result = data.regions || [];
            }
          } catch (e) {
            // ignore — incomplete final chunk
          }
        }
        resolve(result);
      }).catch(reject);
    });
    promise.controller = controller;
    return promise;
  },
  health() { return this._fetch('/health'); },
};
