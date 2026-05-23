const ContextMenu = {
  _visible: false,
  _target: null,

  init() {
    const canvas = CanvasManager.canvas;
    canvas.on('mouse:down', (opt) => this._onMouseDown(opt));

    // Hide context menu on click anywhere else
    document.addEventListener('click', (e) => {
      if (!this._visible) return;
      const menu = document.getElementById('context-menu');
      if (menu && !menu.contains(e.target)) {
        this.close();
      }
    });

    // Hide on escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this._visible) {
        this.close();
      }
    });
  },

  _onMouseDown(opt) {
    if (!opt.e) return;
    const isRightClick = opt.e.button === 2;
    if (!isRightClick) {
      if (this._visible && opt.target) {
        // Left click on canvas while menu is open
        const menu = document.getElementById('context-menu');
        if (menu && !menu.contains(opt.e.target)) {
          this.close();
        }
      }
      return;
    }

    // Check what kind of object is under right click
    const target = opt.target;
    const isIText = target && target._regionId && target.type === 'i-text';
    const isColorBlock = target && target._boxType === 'color-block';
    const isFormula = target && target._isFormula === true;

    if (!isIText && !isColorBlock && !isFormula) {
      this.close();
      return;
    }

    opt.e.preventDefault();
    opt.e.stopPropagation();

    // Select the object if it's not already selected
    const canvas = CanvasManager.canvas;
    const active = canvas.getActiveObject();
    const isInActiveSelection = active && typeof active.getObjects === 'function'
        && active.getObjects().includes(target);
    const isAlreadySelected = active && (isInActiveSelection || active === target);

    if (!isAlreadySelected) {
      canvas.setActiveObject(target);
      canvas.renderAll();
      if (isIText) TextboxManager._syncUI(target);
    }

    this._target = target;
    this.show(opt.e.clientX, opt.e.clientY, target, isIText, isColorBlock, isFormula);
  },

  show(x, y, obj, isIText, isColorBlock, isFormula) {
    const menu = document.getElementById('context-menu');

    // Toggle text-only sections
    const textSections = menu.querySelectorAll('.ctx-text-only');
    textSections.forEach(s => { s.style.display = isIText ? '' : 'none'; });

    if (isIText) {
      document.getElementById('ctx-font-input').value = obj.fontFamily || 'Arial';
      document.getElementById('ctx-font-size').value = obj.fontSize || 24;
      document.getElementById('ctx-color').value = obj.fill || '#000000';
      this._syncBtns();
    }

    menu.style.display = 'block';
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';

    // Ensure menu stays within viewport
    const rect = menu.getBoundingClientRect();
    if (rect.bottom > window.innerHeight) {
      menu.style.top = (y - rect.height) + 'px';
    }
    if (rect.right > window.innerWidth) {
      menu.style.left = (x - rect.width) + 'px';
    }

    this._visible = true;
  },

  close() {
    const menu = document.getElementById('context-menu');
    menu.style.display = 'none';
    document.getElementById('ctx-font-dropdown').style.display = 'none';
    this._visible = false;
    this._target = null;
  },

  // ── Z-Order ──
  orderLayer(method) {
    if (!this._target) return;
    const canvas = CanvasManager.canvas;
    canvas[method](this._target);
    canvas.renderAll();
    if (typeof UndoManager !== 'undefined') UndoManager.saveState('调整图层顺序');
    this.close();
  },

  // ── Delete (handles all object types) ──
  deleteObject() {
    if (!this._target) return;
    const target = this._target;

    if (target._regionId && target.type === 'i-text') {
      TextboxManager.deleteSelected();
    } else {
      const rid = target._regionId;
      CanvasManager.canvas.remove(target);
      if (rid) {
        delete TextboxManager.textObjects[rid];
        TextboxManager.regions = TextboxManager.regions.filter(r => r.id !== rid);
        TextboxManager.updateRegionList();
      }
      CanvasManager.canvas.renderAll();
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('删除对象');
    }
    this.close();
  },

  _syncBtns() {
    const targets = TextboxManager._getTextTargets();
    if (targets.length === 0) return;
    const obj = targets[0];
    document.getElementById('ctx-font-input').value = obj.fontFamily || 'Microsoft YaHei';
    document.getElementById('ctx-font-size').value = obj.fontSize || 24;
    document.getElementById('ctx-color').value = obj.fill || '#000000';

    const btnBold = document.querySelector('#context-menu .ctx-bold');
    const btnItalic = document.querySelector('#context-menu .ctx-italic');
    const btnUnderline = document.querySelector('#context-menu .ctx-underline');
    const btnStrike = document.getElementById('ctx-btn-strike');

    if (btnBold) btnBold.classList.toggle('active', obj.fontWeight === 'bold');
    if (btnItalic) btnItalic.classList.toggle('active', obj.fontStyle === 'italic');
    if (btnUnderline) btnUnderline.classList.toggle('active', obj.underline);
    if (btnStrike) btnStrike.classList.toggle('active', obj.linethrough);
  },
};
