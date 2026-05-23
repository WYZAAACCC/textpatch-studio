const CanvasManager = {
  canvas: null,
  fabricCanvas: null,
  originalImg: null,
  cleanImg: null,
  zoom: 1,
  _showingOriginalBg: false,
  _overlayOriginalImg: null,

  init() {
    const el = document.getElementById('main-canvas');
    const container = document.getElementById('canvas-container');
    const w = container.clientWidth || 1200;
    const h = container.clientHeight || 800;
    el.width = w; el.height = h;

    this.fabricCanvas = new fabric.Canvas('main-canvas', {
      width: w, height: h,
      backgroundColor: '#0a0a16',
      selection: true,
      preserveObjectStacking: true,
      stopContextMenu: true,
      fireRightClick: true,
    });

    this.canvas = this.fabricCanvas;
    this._setupZoom();
    this._setupPan();
    this._setupEvents();
    this._setupDragDrop();
    window.addEventListener('resize', () => this._debouncedResize());
  },

  _resizeTimeout: null,
  _debouncedResize() {
    clearTimeout(this._resizeTimeout);
    this._resizeTimeout = setTimeout(() => this._resize(), 150);
  },

  _resize() {
    const container = document.getElementById('canvas-container');
    const w = container.clientWidth;
    const h = container.clientHeight;
    this.canvas.setWidth(w);
    this.canvas.setHeight(h);
    this.canvas.renderAll();
  },

  _setupZoom() {
    this.canvas.on('mouse:wheel', (opt) => {
      const delta = opt.e.deltaY;
      let newZoom = this.canvas.getZoom();
      newZoom *= delta > 0 ? 0.95 : 1.05;
      newZoom = Math.min(5, Math.max(0.1, newZoom));
      this.zoom = newZoom;
      this.canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, newZoom);
      document.getElementById('zoom-display').textContent = '🔍 ' + Math.round(newZoom * 100) + '%';
      opt.e.preventDefault();
      opt.e.stopPropagation();
      this.canvas.renderAll();
    });
  },

  _setupPan() {
    // Fabric.js v5 only fires mouse:down for left-click (button 0) and
    // right-click with fireRightClick (button 2). Middle-click (button 1)
    // is silently dropped, so we must listen on the native DOM element.
    let isPanning = false, lastX = 0, lastY = 0;
    const el = this.canvas.getElement();
    const parent = el.parentElement;

    parent.addEventListener('mousedown', (e) => {
      if (e.button !== 1) return;
      e.preventDefault();
      e.stopPropagation();
      isPanning = true;
      lastX = e.clientX; lastY = e.clientY;
      this.canvas.selection = false;
      this.canvas.setCursor('grabbing');
    });

    parent.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      e.preventDefault();
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      this.canvas.relativePan({ x: dx, y: dy });
    });

    parent.addEventListener('mouseup', (e) => {
      if (e.button !== 1 || !isPanning) return;
      isPanning = false;
      this.canvas.selection = (typeof App !== 'undefined' && App.currentMode === 'select');
      this.canvas.setCursor('default');
    });

    // Also listen on document for mouseup to handle release outside canvas
    document.addEventListener('mouseup', (e) => {
      if (e.button === 1 && isPanning) {
        isPanning = false;
        this.canvas.selection = (typeof App !== 'undefined' && App.currentMode === 'select');
        this.canvas.setCursor('default');
      }
    });
  },

  _setupEvents() {
    this.canvas.on('selection:created', (e) => TextboxManager.onSelectionChanged(e));
    this.canvas.on('selection:updated', (e) => TextboxManager.onSelectionChanged(e));
    this.canvas.on('selection:cleared', () => TextboxManager.onSelectionCleared());
    this.canvas.on('object:modified', (e) => TextboxManager.onObjectModified(e));
  },

  _setupDragDrop() {
    const container = document.getElementById('canvas-container');
    const overlay = document.getElementById('drop-overlay');

    container.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      overlay.style.display = 'flex';
    });

    container.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.target === container) overlay.style.display = 'none';
    });

    overlay.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });

    overlay.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      overlay.style.display = 'none';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        // Use App's file handler
        const dt = new DataTransfer();
        dt.items.add(file);
        document.getElementById('file-input').files = dt.files;
        App.handleFileSelect({ target: { files: dt.files } });
      } else {
        App.toast('请拖放图片文件', 'warning');
      }
    });

    document.addEventListener('drop', (e) => {
      e.preventDefault();
      overlay.style.display = 'none';
    });
    document.addEventListener('dragover', (e) => e.preventDefault());
  },

  loadImage(url, type) {
    return new Promise((resolve) => {
      fabric.Image.fromURL(url, (img) => {
        if (type === 'original') this.originalImg = img;
        if (type === 'clean') {
          this.cleanImg = img;
          this.setBackground(this.cleanImg);
        }
        resolve(img);
      }, { crossOrigin: 'anonymous' });
    });
  },

  setBackground(img) {
    const objs = this.canvas.getObjects();
    for (const o of objs) {
      if (o._isBackground) { this.canvas.remove(o); break; }
    }
    img._isBackground = true;
    img.selectable = false;
    img.evented = false;
    this.canvas.setWidth(img.width);
    this.canvas.setHeight(img.height);
    this.canvas.add(img);
    this.canvas.sendToBack(img);
    this.canvas.renderAll();
    this._resize();
    document.getElementById('canvas-size-display').textContent = `${img.width}×${img.height}`;
  },

  fitToScreen() {
    if (!this.cleanImg && !this.originalImg) return;
    const img = this.cleanImg || this.originalImg;
    const container = document.getElementById('canvas-container');
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const scale = Math.min(cw / img.width, ch / img.height, 1);
    this.canvas.setZoom(scale);
    this.zoom = scale;
    document.getElementById('zoom-display').textContent = '🔍 ' + Math.round(scale * 100) + '%';
    this.canvas.renderAll();
  },

  // Toggle between original and processed background while editing
  toggleOriginalBg() {
    if (!this.originalImg || !this.cleanImg) return false;

    this._showingOriginalBg = !this._showingOriginalBg;

    // Remove existing background
    const bgObj = this.canvas.getObjects().find(o => o._isBackground);
    if (bgObj) this.canvas.remove(bgObj);

    // Remove the original overlay if exists
    if (this._overlayOriginalImg) {
      this.canvas.remove(this._overlayOriginalImg);
      this._overlayOriginalImg = null;
    }

    if (this._showingOriginalBg) {
      // Show original image as background
      this.originalImg._isBackground = true;
      this.originalImg.selectable = false;
      this.originalImg.evented = false;
      this.canvas.setWidth(this.originalImg.width);
      this.canvas.setHeight(this.originalImg.height);
      this.canvas.add(this.originalImg);
      this.canvas.sendToBack(this.originalImg);
    } else {
      // Show processed/clean image as background
      this.cleanImg._isBackground = true;
      this.cleanImg.selectable = false;
      this.cleanImg.evented = false;
      this.canvas.setWidth(this.cleanImg.width);
      this.canvas.setHeight(this.cleanImg.height);
      this.canvas.add(this.cleanImg);
      this.canvas.sendToBack(this.cleanImg);
    }

    this.canvas.renderAll();
    return this._showingOriginalBg;
  },

  // Check if currently showing original
  isShowingOriginal() {
    return this._showingOriginalBg;
  },

  addRect(params) {
    const rect = new fabric.Rect({
      left: params.left, top: params.top,
      width: params.width, height: params.height,
      fill: 'transparent',
      stroke: params.stroke || '#6c5ce7',
      strokeWidth: 2,
      strokeDashArray: params.dash || [],
      selectable: true,
      hasControls: true,
      hasBorders: true,
      cornerSize: 8,
      cornerColor: '#6c5ce7',
      cornerStrokeColor: '#fff',
      transparentCorners: false,
      _boxType: params.boxType || 'unknown',
    });
    this.canvas.add(rect);
    this.canvas.setActiveObject(rect);
    this.canvas.renderAll();
    return rect;
  },

  removeObject(obj) {
    this.canvas.remove(obj);
    this.canvas.renderAll();
  },

  getCanvasObjects() { return this.canvas.getObjects(); },
  getActiveObject() { return this.canvas.getActiveObject(); },

  setMode(mode) {
    const container = document.getElementById('canvas-container');
    container.classList.remove('mode-select', 'mode-restore', 'mode-detect-auto', 'mode-detect-simple', 'mode-color-block', 'mode-selection');
    container.classList.add('mode-' + mode);

    if (mode === 'select') {
      this.canvas.selection = true;
      this.canvas.getObjects().forEach(o => {
        if (o._boxType) { o.selectable = true; o.evented = true; }
      });
    }
  },
};
