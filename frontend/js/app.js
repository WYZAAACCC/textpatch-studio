const App = {
  currentProjectId: null,
  currentMode: 'select',
  _correctionData: null,
  _correctionEnabled: false,
  _abortController: null,

  async init() {
    CanvasManager.init();
    PostProcess.init();
    FormulaEditor.init();
    Export.init();
    Toolbar.init();
    ContextMenu.init();
    TextboxManager.loadFonts();
    this.setStatus('就绪 — 请导入图片开始');
  },

  // ── Toast Notifications ──
  toast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;

    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    toast.innerHTML = `<span>${icons[type] || ''}</span> ${msg}`;

    container.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 3000);
  },

  // ── Image Import ──
  importImage() {
    document.getElementById('file-input').click();
  },

  async handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      this.toast('请选择图片文件', 'warning');
      return;
    }

    this.setStatus('正在上传...');
    try {
      const proj = await API.uploadImage(file, file.name.replace(/\.[^.]+$/, ''));
      this.currentProjectId = proj.project_id;
      this.setStatus(`项目已创建: ${proj.name} (${proj.width}×${proj.height})`);
      this.toast(`图片已导入 — ${proj.width}×${proj.height}`, 'success');

      await CanvasManager.loadImage(
        API.getImageUrl(proj.project_id, 'original'), 'original'
      );
      await CanvasManager.loadImage(
        API.getImageUrl(proj.project_id, 'original'), 'clean'
      );
      CanvasManager.fitToScreen();
      Toolbar.enableButtons(true);
      this.setMode('select');
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('导入图片');
    } catch (err) {
      this.setStatus('上传失败: ' + err.message);
      this.toast('上传失败: ' + err.message, 'error');
    }
    e.target.value = '';
  },

  // ── Repair Pipeline ──
  async runRepair(mode) {
    if (!this.currentProjectId) return;
    const modeLabel = mode === 'auto' ? 'Auto' : 'Simple';
    this._cancelBtnVisible(true);

    try {
      this.setStatus('正在检测文字区域...');
      await API.detect(this.currentProjectId);

      this.setStatus('正在OCR识别...');
      const ocrResult = await API.ocr(this.currentProjectId);
      const regions = ocrResult.regions || [];

      // ── Integrated AI correction (4C) ──
      // Must run BEFORE approving — correction only targets ocr_done regions
      if (this._correctionEnabled) {
        const eligible = regions.filter(
          r => r.status === 'ocr_done' && r.final_text
        );
        if (eligible.length > 0) {
          this.setStatus('正在AI校正 0/' + eligible.length + '...');
          const promise = API.correctStream(this.currentProjectId, {
            auto_accept: true,
            onStart(total) {
              App.setStatus('正在AI校正 0/' + total + '...');
            },
            onProgress(completed, total) {
              App.setStatus('正在AI校正 ' + completed + '/' + total + '...');
            },
          });
          this._abortController = promise.controller;
          const corrected = await promise;
          this._abortController = null;
          if (corrected) {
            this._correctionData = corrected;
            const changed = corrected.filter(
              r => r.llm && r.llm.changed_chars && r.llm.changed_chars.length > 0
            );
            if (changed.length > 0) {
              this.toast('AI校正完成，' + changed.length + ' 处已自动修正', 'success');
            }
          }
        }
      }

      // Approve all regions for inpainting (correction may have changed statuses)
      const project = await API.getProject(this.currentProjectId);
      for (const r of project.regions) {
        if (r.status !== 'approved') {
          await API.updateRegion(this.currentProjectId, r.id, { status: 'approved' });
        }
      }

      this.setStatus('正在擦除原文字...');
      const simpleMode = mode === 'simple';
      const inpainted = await API.inpaint(this.currentProjectId, simpleMode);

      await CanvasManager.loadImage(
        API.getImageUrl(this.currentProjectId, 'clean_base') + '?t=' + Date.now(), 'clean'
      );
      CanvasManager.fitToScreen();

      const textRegions = inpainted.regions.filter(r => r.final_text || (r.ocr && r.ocr.best_text));
      TextboxManager.createTextObjects(textRegions);

      this._cancelBtnVisible(false);
      this.hideLoading();
      this.setStatus(`${modeLabel}修复完成 — ${textRegions.length} 个文字区域可编辑`);
      this.toast(`${modeLabel}修复完成，共 ${textRegions.length} 个文字区域`, 'success');
      this.setMode('select');
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('修复完成');
    } catch (err) {
      this._abortController = null;
      this._cancelBtnVisible(false);
      this.hideLoading();
      if (err.name === 'AbortError') {
        this.setStatus('操作已取消');
        this.toast('操作已取消', 'warning');
      } else {
        this.setStatus('修复失败: ' + err.message);
        this.toast('修复失败: ' + err.message, 'error');
      }
    }
  },

  // ── Abort current operation (4D) ──
  abortOperation() {
    if (this._abortController) {
      this._abortController.abort();
    }
  },

  _cancelBtnVisible(show) {
    const btn = document.getElementById('btn-cancel-operation');
    if (btn) btn.style.display = show ? '' : 'none';
  },

  // ── LLM Auto-Correction ──
  async runLLMCorrection() {
    if (!this.currentProjectId) return;
    this.showLoading('正在进行AI校正...');
    this._cancelBtnVisible(true);
    this.setStatus('正在调用LLM校正文字...');
    try {
      const promise = API.correctStream(this.currentProjectId, {
        auto_accept: false,
        onStart(total) {
          App.setStatus('正在AI校正 0/' + total + '...');
        },
        onProgress(completed, total) {
          App.setStatus('正在AI校正 ' + completed + '/' + total + '...');
        },
      });
      this._abortController = promise.controller;
      const result = await promise;
      this._abortController = null;
      this._cancelBtnVisible(false);
      this.hideLoading();
      this._correctionData = result || [];
      this.showCorrectionDiff();
    } catch (err) {
      this._abortController = null;
      this._cancelBtnVisible(false);
      this.hideLoading();
      if (err.name === 'AbortError') {
        this.setStatus('操作已取消');
        this.toast('操作已取消', 'warning');
      } else {
        this.setStatus('AI校正失败: ' + err.message);
        this.toast('AI校正失败: ' + err.message, 'error');
      }
    }
  },

  showCorrectionDiff() {
    const regions = this._correctionData;
    const changed = regions.filter(r => r.llm && r.llm.changed_chars && r.llm.changed_chars.length > 0);

    const body = document.getElementById('correction-body');
    if (changed.length === 0) {
      body.innerHTML = '<p class="correction-empty">没有需要校正的文字 — 所有文字已准确识别</p>';
      document.getElementById('btn-accept-all-corrections').style.display = 'none';
    } else {
      body.innerHTML = `<p class="correction-summary">发现 ${changed.length} 处需要校正的文字:</p>` +
        changed.map(r => {
          const original = r.final_text || r.ocr?.best_text || '';
          const suggested = r.llm.suggested_text || original;
          const changedChars = r.llm.changed_chars || [];
          return `<div class="diff-region-row" id="diff-row-${r.id}">
            <div class="diff-rid">${r.id.slice(-6)}</div>
            <div class="diff-texts">
              <div class="diff-original" title="原始文字">${this._escapeHtml(original)}</div>
              <div class="diff-arrow">→</div>
              <div class="diff-corrected" title="AI建议">${this._escapeHtml(suggested)}</div>
            </div>
            <div class="diff-actions">
              <button class="btn-accept" onclick="App.acceptCorrection('${r.id}')">接受</button>
              <button class="btn-reject" onclick="App.rejectCorrection('${r.id}')">拒绝</button>
            </div>
          </div>`;
        }).join('');
      document.getElementById('btn-accept-all-corrections').style.display = '';
    }

    document.getElementById('overlay-correction').style.display = 'flex';
  },

  acceptCorrection(regionId) {
    const region = this._correctionData.find(r => r.id === regionId);
    if (!region || !region.llm) return;

    const suggestedText = region.llm.suggested_text;
    const obj = TextboxManager.textObjects[regionId];
    if (obj && suggestedText) {
      obj.set('text', suggestedText);
      TextboxManager.canvas().renderAll();
      TextboxManager.onObjectModified({ target: obj });
    }

    // Update backend
    API.updateRegion(this.currentProjectId, regionId, {
      final_text: suggestedText, status: 'approved',
    }).catch(e => console.warn('Accept correction sync failed:', e));

    const row = document.getElementById('diff-row-' + regionId);
    if (row) row.style.display = 'none';

    this.toast('已接受校正 ' + regionId.slice(-6), 'success');
    if (typeof UndoManager !== 'undefined') UndoManager.saveState('接受LLM校正');
    this._checkAllResolved();
  },

  rejectCorrection(regionId) {
    const row = document.getElementById('diff-row-' + regionId);
    if (row) row.style.display = 'none';
    this.toast('已拒绝校正', 'info');
    this._checkAllResolved();
  },

  acceptAllCorrections() {
    const changed = this._correctionData.filter(r => r.llm && r.llm.changed_chars && r.llm.changed_chars.length > 0);
    for (const r of changed) this.acceptCorrection(r.id);
    this.closeCorrection();
  },

  closeCorrection() {
    document.getElementById('overlay-correction').style.display = 'none';
    this._correctionData = null;
  },

  _checkAllResolved() {
    const visible = document.querySelectorAll('#correction-body .diff-region-row[style*="display: none"], #correction-body .diff-region-row:not([style])');
    const all = document.querySelectorAll('#correction-body .diff-region-row');
    if (visible.length === 0 && all.length > 0) {
      this.closeCorrection();
      this.toast('所有校正已完成', 'success');
    }
  },

  _escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  },

  // ── Background Toggle Comparison ──
  toggleCompareBg() {
    if (!this.currentProjectId) return;
    if (!CanvasManager.originalImg || !CanvasManager.cleanImg) {
      this.toast('请先导入并处理图片', 'warning');
      return;
    }

    const showing = CanvasManager.toggleOriginalBg();
    const btn = document.getElementById('btn-compare-bg');
    if (showing) {
      this.setStatus('背景对比 — 正在查看原图（再次点击/按Tab返回）');
      if (btn) btn.classList.add('active');
    } else {
      this.setStatus('背景对比已关闭 — 返回处理图');
      if (btn) btn.classList.remove('active');
    }
  },

  // Side-by-side: right panel showing original image (non-blocking)
  _compareDragging: false,

  toggleCompareSideBySide() {
    const panel = document.getElementById('compare-panel');
    const btn = document.getElementById('btn-compare-side');
    if (panel.style.display === 'flex') {
      panel.style.display = 'none';
      document.removeEventListener('mousemove', this._onComparePanelResize);
      document.removeEventListener('mouseup', this._onComparePanelResizeUp);
      this.setStatus('参考图已关闭');
      if (btn) btn.classList.remove('active');
      return;
    }
    if (!this.currentProjectId) return;

    document.getElementById('compare-ref-img').src = API.getImageUrl(this.currentProjectId, 'original');
    panel.style.display = 'flex';
    if (btn) btn.classList.add('active');

    // Resize from left edge
    const handle = document.getElementById('compare-panel-resize');
    handle.onmousedown = (e) => {
      this._compareDragging = true;
      e.preventDefault();
    };

    this._onComparePanelResize = (e) => {
      if (!this._compareDragging) return;
      const newWidth = window.innerWidth - e.clientX;
      panel.style.width = Math.max(200, Math.min(window.innerWidth * 0.5, newWidth)) + 'px';
    };
    this._onComparePanelResizeUp = () => {
      this._compareDragging = false;
    };
    document.addEventListener('mousemove', this._onComparePanelResize);
    document.addEventListener('mouseup', this._onComparePanelResizeUp);

    this.setStatus('参考图已打开 — 原图显示在右侧面板');
    this.toast('原图参考面板已打开，可拖拽左边缘调整大小', 'info');
  },

  // ── Mode Switching ──
  setMode(mode) {
    this.currentMode = mode;
    Toolbar.setMode(mode);
    CanvasManager.setMode(mode);

    if (mode === 'select') {
      CanvasManager.canvas.selection = true;
    }
  },

  // ── Export ──
  openExport() { Export.open(); },
  closeExport() { Export.close(); },
  async doExport() { await Export.doExport(); },

  // ── Status Bar ──
  setStatus(msg) {
    document.getElementById('status-text').textContent = msg;
    console.log('[Status]', msg);
  },

  // ── Loading Overlay ──
  showLoading(msg) {
    document.getElementById('loading-text').textContent = msg;
    document.getElementById('overlay-loading').style.display = 'flex';
  },

  hideLoading() {
    document.getElementById('overlay-loading').style.display = 'none';
  },
};

document.addEventListener('DOMContentLoaded', () => App.init());
