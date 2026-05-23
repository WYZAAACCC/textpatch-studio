const PostProcess = {
  drawingBox: null,
  startX: 0,
  startY: 0,
  isDrawing: false,

  init() {
    const canvas = CanvasManager.canvas;
    canvas.on('mouse:down', (opt) => this._onMouseDown(opt));
    canvas.on('mouse:move', (opt) => this._onMouseMove(opt));
    canvas.on('mouse:up', (opt) => this._onMouseUp(opt));
    // Show color picker when clicking on an existing color block (edit mode)
    canvas.on('mouse:down', (opt) => {
      if (opt.target && opt.target._boxType === 'color-block') {
        this._editColorBlock(opt.target);
      }
    });
  },

  _onMouseDown(opt) {
    if (!['restore', 'detect-auto', 'detect-simple', 'selection', 'color-block'].includes(App.currentMode)) return;
    // Don't start drawing if clicking on an existing object (text box, formula, color block)
    // This allows dragging/moving existing objects without creating tool boxes
    if (opt.target && (opt.target._regionId || opt.target._boxType || opt.target._isFormula)) return;
    // Also check subTargets (Fabric.js v5) — catches clicks on control handles where
    // opt.target may be null but the parent object is still in subTargets
    if (opt.subTargets && opt.subTargets.some(o => o._regionId || o._boxType || o._isFormula)) return;
    const pointer = CanvasManager.canvas.getPointer(opt.e);
    this.startX = pointer.x;
    this.startY = pointer.y;
    this.isDrawing = true;

    const colors = {
      'restore': '#00d2a0',
      'detect-auto': '#6c5ce7',
      'detect-simple': '#f0a040',
      'selection': '#ffd700',
      'color-block': '#ff6b6b',
    };

    const fills = {
      'restore': 'rgba(0,210,160,0.06)',
      'detect-auto': 'rgba(108,92,231,0.06)',
      'detect-simple': 'rgba(240,160,64,0.06)',
      'selection': 'rgba(255,215,0,0.08)',
      'color-block': 'rgba(255,107,107,0.25)',
    };

    const dashPatterns = {
      'selection': [8, 4],
    };

    this.drawingBox = new fabric.Rect({
      left: this.startX, top: this.startY,
      width: 1, height: 1,
      fill: fills[App.currentMode] || 'rgba(255,255,255,0.05)',
      stroke: colors[App.currentMode] || '#fff',
      strokeWidth: 2.5,
      strokeDashArray: dashPatterns[App.currentMode] || [],
      selectable: false,
      evented: false,
      rx: 2, ry: 2,
    });
    CanvasManager.canvas.add(this.drawingBox);
  },

  _onMouseMove(opt) {
    if (!this.isDrawing || !this.drawingBox) return;
    const pointer = CanvasManager.canvas.getPointer(opt.e);
    const w = pointer.x - this.startX;
    const h = pointer.y - this.startY;
    this.drawingBox.set({
      left: w > 0 ? this.startX : pointer.x,
      top: h > 0 ? this.startY : pointer.y,
      width: Math.abs(w),
      height: Math.abs(h),
    });
    CanvasManager.canvas.renderAll();
  },

  _onMouseUp(opt) {
    if (!this.isDrawing) return;
    this.isDrawing = false;

    if (!this.drawingBox || this.drawingBox.width < 5 || this.drawingBox.height < 5) {
      if (this.drawingBox) CanvasManager.removeObject(this.drawingBox);
      this.drawingBox = null;
      return;
    }

    const box = this.drawingBox;
    const rect = {
      x: Math.round(box.left),
      y: Math.round(box.top),
      width: Math.round(box.width),
      height: Math.round(box.height),
    };
    this.drawingBox = null;

    switch (App.currentMode) {
      case 'restore': this._processRestore(box, rect); break;
      case 'detect-auto': this._processDetect(box, rect, 'auto'); break;
      case 'detect-simple': this._processDetect(box, rect, 'simple'); break;
      case 'selection': this._processSelection(box, rect); break;
      case 'color-block': this._processColorBlock(box, rect); break;
    }
  },

  async _processRestore(box, rect) {
    App.showLoading('正在恢复原始像素...');
    CanvasManager.removeObject(box);
    try {
      await API.restoreRegion(App.currentProjectId, rect.x, rect.y, rect.width, rect.height);
      await CanvasManager.loadImage(
        API.getImageUrl(App.currentProjectId, 'clean_base') + '?t=' + Date.now(), 'clean'
      );
      CanvasManager.fitToScreen();
      if (typeof UndoManager !== 'undefined') {
        UndoManager.saveState('恢复像素');
      }
      App.setStatus('恢复完成');
      App.hideLoading();
      App.toast('恢复完成 — 框内像素已还原为原始图像', 'success');
    } catch (e) {
      App.setStatus('恢复失败: ' + e.message);
      App.hideLoading();
      App.toast('恢复失败: ' + e.message, 'error');
    }
  },

  async _processDetect(box, rect, mode) {
    const modeLabel = mode === 'auto' ? 'Auto' : 'Simple';
    App.showLoading(`正在${modeLabel}检测并擦除...`);
    CanvasManager.removeObject(box);
    try {
      const proj = await API.detectRegion(App.currentProjectId, rect.x, rect.y, rect.width, rect.height, mode);

      // Find truly new regions by comparing against existing ones (ignore slice hack)
      const existingIds = new Set(TextboxManager.regions.map(r => r.id));
      const actuallyNew = (proj.regions || []).filter(r => !existingIds.has(r.id));

      // Remove all old text boxes that overlap with the inpainted area.
      // The backend always erases text inside the detection box, so old
      // IText objects must be cleared regardless of whether new regions
      // were detected — otherwise stale text overlays the clean background.
      const boxRect = { left: rect.x, top: rect.y, width: rect.width, height: rect.height };
      const removedIds = [];
      for (const [rid, obj] of Object.entries(TextboxManager.textObjects)) {
        const cx = obj.left + (obj.width || 0) * (obj.scaleX || 1) / 2;
        const cy = obj.top + (obj.height || 0) * (obj.scaleY || 1) / 2;
        if (cx >= boxRect.left && cx <= boxRect.left + boxRect.width &&
            cy >= boxRect.top && cy <= boxRect.top + boxRect.height) {
          CanvasManager.canvas.remove(obj);
          delete TextboxManager.textObjects[rid];
          removedIds.push(rid);
        }
      }
      if (removedIds.length > 0) {
        TextboxManager.regions = TextboxManager.regions.filter(r => !removedIds.includes(r.id));
      }

      TextboxManager.addRegions(actuallyNew);
      await CanvasManager.loadImage(
        API.getImageUrl(App.currentProjectId, 'clean_base') + '?t=' + Date.now(), 'clean'
      );
      CanvasManager.fitToScreen();
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('补识别');
      const resultMsg = actuallyNew.length > 0
        ? `${modeLabel}检测完成 — 发现 ${actuallyNew.length} 个新文字区域`
        : `${modeLabel}擦除完成 — 框内文字已清除`;
      App.setStatus(resultMsg);
      App.hideLoading();
      App.toast(resultMsg, 'success');
    } catch (e) {
      App.setStatus('检测失败: ' + e.message);
      App.hideLoading();
      App.toast('检测失败: ' + e.message, 'error');
    }
  },

  _processSelection(box, rect) {
    const count = TextboxManager.selectAllInRect(rect);
    CanvasManager.removeObject(box);
    App.setMode('select');
    if (count > 0) {
      App.toast(`已选取 ${count} 个文字框，可批量修改格式`, 'info');
    } else {
      App.toast('框内无文字框', 'warning');
    }
  },

  // ── Color Block Overlay ──
  _pendingColorBlock: null,
  _pendingColorRect: null,

  _processColorBlock(box, rect) {
    CanvasManager.removeObject(box);
    this._pendingColorBlock = null;
    this._pendingColorRect = rect;
    const popup = document.getElementById('color-picker-popup');
    const canvasRect = document.getElementById('canvas-container').getBoundingClientRect();
    popup.style.left = (canvasRect.left + rect.x + rect.width / 2 - 100) + 'px';
    popup.style.top = (canvasRect.top + rect.y + rect.height + 10) + 'px';
    popup.style.display = 'block';
    document.getElementById('color-picker-input').value = '#ff6b6b';
    document.getElementById('color-opacity-slider').value = 85;
    document.getElementById('opacity-val').textContent = '85%';
    this._buildColorSwatches();
  },

  // Show color picker for editing an existing color block
  _editColorBlock(block) {
    this._pendingColorBlock = block;
    this._pendingColorRect = null;
    const popup = document.getElementById('color-picker-popup');
    const canvasRect = document.getElementById('canvas-container').getBoundingClientRect();
    const vpt = CanvasManager.canvas.viewportTransform;
    const zoom = vpt ? vpt[0] : 1;
    const centerX = (block.left + (block.width * (block.scaleX || 1)) / 2) * zoom + (vpt ? vpt[4] : 0);
    const bottomY = (block.top + (block.height * (block.scaleY || 1)) + 10) * zoom + (vpt ? vpt[5] : 0);
    popup.style.left = (canvasRect.left + centerX - 100) + 'px';
    popup.style.top = (canvasRect.top + bottomY) + 'px';
    popup.style.display = 'block';
    document.getElementById('color-picker-input').value = block.fill || '#ff6b6b';
    const opacityPct = Math.round((block.opacity != null ? block.opacity : 0.85) * 100);
    document.getElementById('color-opacity-slider').value = opacityPct;
    document.getElementById('opacity-val').textContent = opacityPct + '%';
    this._buildColorSwatches();
  },

  hideColorPicker() {
    document.getElementById('color-picker-popup').style.display = 'none';
    this._pendingColorBlock = null;
    this._pendingColorRect = null;
  },

  confirmColorBlock() {
    const color = document.getElementById('color-picker-input').value;
    const opacity = parseInt(document.getElementById('color-opacity-slider').value, 10) / 100;
    const existingBlock = this._pendingColorBlock;
    const rect = this._pendingColorRect;
    document.getElementById('color-picker-popup').style.display = 'none';

    if (existingBlock) {
      existingBlock.set({ fill: color, stroke: color, opacity: opacity });
      CanvasManager.canvas.renderAll();
      App.toast('色块已更新', 'success');
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('修改色块');
      this._pendingColorBlock = null;
      return;
    }

    if (!rect) return;

    const block = new fabric.Rect({
      left: rect.x, top: rect.y,
      width: rect.width, height: rect.height,
      fill: color,
      stroke: color,
      strokeWidth: 0,
      opacity: opacity,
      _boxType: 'color-block',
      selectable: true,
      hasControls: true,
      hasBorders: true,
      cornerSize: 8,
      cornerColor: '#ff6b6b',
      cornerStrokeColor: '#fff',
      transparentCorners: false,
      borderColor: '#ff6b6b',
      rx: 2, ry: 2,
    });

    CanvasManager.canvas.add(block);
    CanvasManager.canvas.setActiveObject(block);
    CanvasManager.canvas.renderAll();
    App.toast('色块已创建 — 可拖动调整位置和大小', 'success');
    if (typeof UndoManager !== 'undefined') UndoManager.saveState('创建色块');

    this._pendingColorBlock = null;
    this._pendingColorRect = null;
  },

  _buildColorSwatches() {
    const swatches = document.getElementById('color-swatches');
    const colors = [
      '#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#000000',
      '#ff6b6b', '#ff922b', '#ffd43b', '#69db7c', '#4dabf7', '#6c5ce7',
      '#fa5252', '#e64980', '#f783ac', '#faa2c1', '#e599f7', '#845ef7',
      '#5c7cfa', '#339af0', '#22b8cf', '#20c997', '#51cf66', '#82c91e',
      '#e8590c', '#fd7e14', '#fab005', '#94d82d', '#38d9a9', '#748ffc',
    ];
    swatches.innerHTML = colors.map(c =>
      `<div class="color-swatch" style="background:${c}" onclick="PostProcess.selectColor('${c}')" title="${c}"></div>`
    ).join('');
  },

  selectColor(color) {
    document.getElementById('color-picker-input').value = color;
  },

  async pickScreenColor() {
    try {
      if (!window.EyeDropper) {
        App.toast('当前浏览器不支持取色管，请使用Chrome/Edge', 'warning');
        return;
      }
      const eyeDropper = new EyeDropper();
      const result = await eyeDropper.open();
      document.getElementById('color-picker-input').value = result.sRGBHex;
      App.toast(`已取色: ${result.sRGBHex}`, 'info');
    } catch (e) {
      // AbortError = user cancelled, silently ignore
      if (e.name !== 'AbortError') {
        App.toast('取色失败: ' + (e.message || '未知错误'), 'error');
      }
    }
  },

};
