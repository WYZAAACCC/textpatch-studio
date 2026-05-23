const UndoManager = {
  historyStack: [],
  historyIndex: -1,
  maxStates: 50,
  _suppressNext: false,
  _restoring: false,

  saveState(label, extra) {
    if (this._suppressNext) { this._suppressNext = false; return; }
    if (this._restoring) return;

    const canvas = CanvasManager.canvas;
    const json = canvas.toJSON([
      '_regionId', '_boxType', '_isFormula', '_latexSource',
      '_isBackground', '_colorBlock', '_linethrough',
    ]);

    if (json.objects) {
      json.objects = json.objects.filter(o => !o._isBackground);
    }

    const useOriginal = CanvasManager.isShowingOriginal();
    const bgInfo = {
      showingOriginal: useOriginal,
      width: CanvasManager.cleanImg ? CanvasManager.cleanImg.width : 0,
      height: CanvasManager.cleanImg ? CanvasManager.cleanImg.height : 0,
      _bgImg: useOriginal ? CanvasManager.originalImg : CanvasManager.cleanImg,
    };

    // Extra can override _bgImg with a specifically captured pre-operation image
    if (extra && extra.savedCleanImg) {
      bgInfo._bgImg = extra.savedCleanImg;
    }

    this.historyStack = this.historyStack.slice(0, this.historyIndex + 1);
    this.historyStack.push({ label, state: JSON.stringify(json), bgInfo });

    if (this.historyStack.length > this.maxStates) {
      this.historyStack.shift();
    }
    this.historyIndex = this.historyStack.length - 1;
  },

  _suppressNextSave() {
    this._suppressNext = true;
  },

  undo() {
    if (this.historyIndex <= 0) {
      App.toast('无法继续撤销', 'warning');
      return false;
    }
    this.historyIndex--;
    return this._restoreState(this.historyStack[this.historyIndex]);
  },

  redo() {
    if (this.historyIndex >= this.historyStack.length - 1) {
      App.toast('无法继续重做', 'warning');
      return false;
    }
    this.historyIndex++;
    return this._restoreState(this.historyStack[this.historyIndex]);
  },

  _restoreState(entry) {
    this._restoring = true;
    const canvas = CanvasManager.canvas;
    const json = JSON.parse(entry.state);
    const bgInfo = entry.bgInfo || {};
    const useOriginal = bgInfo.showingOriginal;
    const savedBgImg = bgInfo._bgImg || null;

    const doRestore = (bgImg) => {
      canvas.loadFromJSON(json, () => {
        if (bgImg) {
          canvas.setWidth(bgImg.width);
          canvas.setHeight(bgImg.height);
          bgImg._isBackground = true;
          bgImg.selectable = false;
          bgImg.evented = false;
          canvas.add(bgImg);
          canvas.sendToBack(bgImg);
          canvas.renderAll();
        }

        if (typeof TextboxManager !== 'undefined') TextboxManager._rebuildTextObjectsMap();

        if (CanvasManager._showingOriginalBg !== useOriginal && CanvasManager.originalImg && CanvasManager.cleanImg) {
          CanvasManager._showingOriginalBg = useOriginal;
        }

        canvas.renderAll();
        const isRedo = this.historyIndex > this.historyStack.indexOf(entry);
        App.toast(`${isRedo ? '已重做' : '已撤销'}: ${entry.label}`, 'info');
        this._restoring = false;
      });
    };

    if (savedBgImg) {
      if (!useOriginal) CanvasManager.cleanImg = savedBgImg;
      doRestore(savedBgImg);
    } else {
      const bgImg = useOriginal ? CanvasManager.originalImg : CanvasManager.cleanImg;
      doRestore(bgImg);
    }

    return true;
  },
};
