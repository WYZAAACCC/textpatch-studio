const Toolbar = {
  init() {
    document.addEventListener('keydown', (e) => this._onKeyDown(e));
  },

  setMode(mode) {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`[data-mode="${mode}"]`);
    if (btn) btn.classList.add('active');

    const modeNames = {
      'select': '🖱 选择',
      'restore': '🔄 恢复框',
      'detect-auto': '🔍 补识别-Auto',
      'detect-simple': '🔎 补识别-Simple',
      'color-block': '🟥 色块覆盖',
      'selection': '⊞ 选取框',
    };
    document.getElementById('mode-display').textContent = modeNames[mode] || mode;

    // Show/hide mode indicator
    const indicator = document.getElementById('mode-indicator');
    if (mode !== 'select') {
      indicator.textContent = (modeNames[mode] || mode).replace(/[^一-鿿\w\s-]/g, '');
      indicator.style.display = 'block';
      clearTimeout(this._indicatorTimeout);
      this._indicatorTimeout = setTimeout(() => {
        indicator.style.display = 'none';
      }, 2000);
    } else {
      indicator.style.display = 'none';
    }

    // Font controls only in select mode with active selection
    const fc = document.getElementById('font-controls');
    if (mode === 'select') {
      const active = CanvasManager.canvas.getActiveObject();
      fc.style.display = (active && active._regionId) ? 'flex' : 'none';
    } else {
      fc.style.display = 'none';
    }
  },

  enableButtons(enable) {
    ['btn-auto-repair', 'btn-simple-repair', 'btn-ai-correct', 'btn-compare-bg', 'btn-compare-side', 'btn-export'].forEach(id => {
      document.getElementById(id).disabled = !enable;
    });
  },

  _onKeyDown(e) {
    // Don't intercept when typing in inputs or textareas
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    // Don't intercept when editing IText
    if (CanvasManager.canvas.getActiveObject()?.isEditing) return;

    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;

    // Ctrl+Z: Undo
    if (ctrl && key === 'z' && !e.shiftKey) { e.preventDefault(); UndoManager.undo(); return; }
    // Ctrl+Y or Ctrl+Shift+Z: Redo
    if ((ctrl && key === 'y') || (ctrl && e.shiftKey && key === 'z')) { e.preventDefault(); UndoManager.redo(); return; }
    // Ctrl+C: Copy
    if (ctrl && key === 'c') { e.preventDefault(); ClipboardManager.copy(); return; }
    // Ctrl+V: Paste
    if (ctrl && key === 'v') { e.preventDefault(); ClipboardManager.paste(); return; }
    // Ctrl+O: Import
    if (ctrl && key === 'o') { e.preventDefault(); App.importImage(); return; }
    // Ctrl+E: Export
    if (ctrl && key === 'e') { e.preventDefault(); App.openExport(); return; }
    // Ctrl+Q: Compare toggle (live background swap)
    if (ctrl && key === 'q') { e.preventDefault(); App.toggleCompareBg(); return; }
    // Ctrl+Shift+Q: Side-by-side comparison overlay
    if (ctrl && e.shiftKey && key === 'q') { e.preventDefault(); App.toggleCompareSideBySide(); return; }
    // Ctrl+B: Bold
    if (ctrl && key === 'b') { e.preventDefault(); TextboxManager.toggleBold(); return; }
    // Ctrl+I: Italic
    if (ctrl && key === 'i') { e.preventDefault(); TextboxManager.toggleItalic(); return; }
    // Ctrl+U: Underline
    if (ctrl && key === 'u') { e.preventDefault(); TextboxManager.toggleUnderline(); return; }
    // Ctrl+A: Select all text boxes
    if (ctrl && key === 'a') { e.preventDefault(); TextboxManager.selectAllTextboxes(); return; }

    // Mode shortcuts (no Ctrl)
    if (key === 'v' && !ctrl && App.currentProjectId) { App.setMode('select'); return; }
    if (key === 'r' && !ctrl && App.currentProjectId) { App.setMode('restore'); return; }
    if (key === 'a' && !ctrl && App.currentProjectId) { App.setMode('detect-auto'); return; }
    if (key === 's' && !ctrl && App.currentProjectId) { App.setMode('detect-simple'); return; }
    if (key === 'b' && !ctrl && App.currentProjectId) { App.setMode('color-block'); return; }
    if (key === 'm' && !ctrl && App.currentProjectId) { App.setMode('selection'); return; }

    // Delete / Backspace: Delete selected
    if ((key === 'delete' || key === 'backspace') && !ctrl) {
      const active = CanvasManager.canvas.getActiveObject();
      if (active && (active._regionId || active._boxType)) {
        e.preventDefault();
        if (active._regionId) TextboxManager.deleteSelected();
        else { CanvasManager.canvas.remove(active); CanvasManager.canvas.renderAll(); }
      }
      return;
    }

    // Escape: Deselect
    if (key === 'escape') {
      TextboxManager.deselectAll();
      // Also hide color picker popup if open
      if (typeof PostProcess !== 'undefined') PostProcess.hideColorPicker();
      // If showing original bg, switch back
      if (CanvasManager.isShowingOriginal()) App.toggleCompareBg();
      return;
    }

    // Tab: Quick-peek original image (toggle)
    if (key === 'tab' && !ctrl && App.currentProjectId) {
      e.preventDefault();
      App.toggleCompareBg();
      return;
    }
  },
};
