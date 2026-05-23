const ClipboardManager = {
  clipboard: null,
  _pasteOffset: { x: 30, y: 30 },

  copy() {
    const canvas = CanvasManager.canvas;
    const active = canvas.getActiveObject();

    if (!active) { App.toast('请先选择要复制的对象', 'warning'); return; }

    // Handle ActiveSelection (multi-select)
    if (typeof active.getObjects === 'function') {
      const objects = active.getObjects();
      this.clipboard = objects.map(o => {
        const serialized = o.toObject([
          '_regionId', '_boxType', '_isFormula', '_latexSource', '_colorBlock',
        ]);
        return JSON.parse(JSON.stringify(serialized));
      });
      this.clipboard._isMulti = true;
    } else if (active._regionId || active._isFormula || active._boxType) {
      const serialized = active.toObject([
        '_regionId', '_boxType', '_isFormula', '_latexSource', '_colorBlock',
      ]);
      this.clipboard = [JSON.parse(JSON.stringify(serialized))];
      this.clipboard._isMulti = false;
    } else {
      App.toast('无法复制此对象', 'warning');
      return;
    }

    this._pasteOffset = { x: 30, y: 30 };
    App.toast('已复制', 'info');
  },

  paste() {
    if (!this.clipboard) { App.toast('剪贴板为空', 'warning'); return; }

    const canvas = CanvasManager.canvas;
    const offsetX = this._pasteOffset.x;
    const offsetY = this._pasteOffset.y;
    this._pasteOffset.x += 30;
    this._pasteOffset.y += 30;
    if (this._pasteOffset.x > 200) this._pasteOffset = { x: 30, y: 30 };

    fabric.util.enlivenObjects(this.clipboard.filter(o => typeof o !== 'string'), (objects) => {
      for (const obj of objects) {
        obj.set({
          left: (obj.left || 0) + offsetX,
          top: (obj.top || 0) + offsetY,
          evented: true,
          selectable: true,
        });

        if (obj._regionId) {
          const newId = 'region_paste_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
          obj._regionId = newId;

          if (obj.type === 'i-text') {
            TextboxManager.textObjects[newId] = obj;
            const bbox = [obj.left, obj.top, obj.left + (obj.width || 100) * (obj.scaleX || 1), obj.top + (obj.height || 24) * (obj.scaleY || 1)];
            TextboxManager.regions.push({
              id: newId, bbox, final_text: obj.text || '',
              style: {
                font_family: obj.fontFamily, font_size: obj.fontSize, font_weight: obj.fontWeight,
                italic: obj.fontStyle === 'italic', underline: obj.underline, align: obj.textAlign,
                color: TextboxManager._parseColor(obj.fill),
              },
            });
          }
        }

        if (obj._isFormula) {
          obj._regionId = 'formula_paste_' + Date.now();
          if (obj._latexSource) {
            // Re-render formula if source available
            const src = obj._latexSource;
            const tempDiv = document.createElement('div');
            tempDiv.style.position = 'absolute';
            tempDiv.style.left = '-9999px';
            document.body.appendChild(tempDiv);
            try {
              katex.render(src, tempDiv, { throwOnError: false, displayMode: false, trust: true, output: 'html' });
              const svgEl = tempDiv.querySelector('svg');
              if (svgEl) {
                const svgData = new XMLSerializer().serializeToString(svgEl);
                const svgBase64 = btoa(unescape(encodeURIComponent(svgData)));
                obj.setSrc('data:image/svg+xml;base64,' + svgBase64, () => canvas.renderAll());
              }
            } catch (e) { /* keep original image */ }
            document.body.removeChild(tempDiv);
          }
        }

        canvas.add(obj);
      }

      if (objects.length === 1) {
        canvas.setActiveObject(objects[0]);
      } else if (objects.length > 1) {
        const sel = new fabric.ActiveSelection(objects, { canvas });
        canvas.setActiveObject(sel);
      }

      canvas.renderAll();
      TextboxManager.updateRegionList();
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('粘贴');
      App.toast(`已粘贴 ${objects.length} 个对象`, 'info');
    });
  },
};
