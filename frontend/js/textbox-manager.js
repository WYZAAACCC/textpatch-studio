const TextboxManager = {
  regions: [],
  textObjects: {},
  currentBold: false,
  currentItalic: false,
  currentUnderline: false,
  currentStrikethrough: false,
  allFonts: [],
  currentFont: 'Microsoft YaHei',
  _fontSearchTimer: null,

  // ── Prioritized common fonts (shown at top) ──
  PRIORITY_FONTS: [
    'SimHei',     // 黑体
    'SimSun',     // 宋体
    'STSong',     // 华文宋体/小标宋
    'FangSong',   // 仿宋
    'KaiTi',      // 楷体
    'STKaiti',    // 华文楷体
    'Microsoft YaHei', // 微软雅黑
    'SimSun-ExtB', // 宋体扩展
    'DFKai-SB',   // 标楷体
    'NSimSun',    // 新宋体
    'FZShuTi',    // 书体
    'FZYaoti',    // 姚体
    'STFangsong', // 华文仿宋
    'STCaiyun',   // 华文彩云
    'STXihei',    // 华文细黑
    'STXingkai',  // 华文行楷
    'STXinwei',   // 华文新魏
    'STLiti',     // 华文隶书
    'STHupo',     // 华文琥珀
    'Arial',
    'Times New Roman',
    'Times',
    'Georgia',
    'Verdana',
    'Courier New',
    'Tahoma',
    'Trebuchet MS',
    'Palatino Linotype',
    'Comic Sans MS',
    'Impact',
    'Lucida Console',
  ],

  // ── Font Loading ──
  async loadFonts() {
    try {
      const data = await API.getFonts();
      let rawFonts = (data.fonts || []).map(f => f.name || f);
      // Sort into priority fonts first, then rest alphabetically
      const priority = [];
      const rest = [];
      for (const f of rawFonts) {
        if (this.PRIORITY_FONTS.some(p => p.toLowerCase() === f.toLowerCase())) {
          priority.push(f);
        } else {
          rest.push(f);
        }
      }
      priority.sort((a, b) => {
        const ia = this.PRIORITY_FONTS.findIndex(p => p.toLowerCase() === a.toLowerCase());
        const ib = this.PRIORITY_FONTS.findIndex(p => p.toLowerCase() === b.toLowerCase());
        if (ia >= 0 && ib >= 0) return ia - ib;
        if (ia >= 0) return -1;
        if (ib >= 0) return 1;
        return a.localeCompare(b);
      });
      rest.sort((a, b) => a.localeCompare(b));
      this.allFonts = [...priority, ...rest];
    } catch (e) {
      console.warn('Font loading failed, using defaults:', e);
      this.allFonts = [...this.PRIORITY_FONTS];
    }
  },

  // ── Helper: get text objects from selection (handles multi-select) ──
  _getTextTargets() {
    const active = this.canvas().getActiveObject();
    if (!active) return [];
    // Use duck-typing for ActiveSelection (type can be 'activeselection' or 'activeSelection')
    if (typeof active.getObjects === 'function') {
      return active.getObjects().filter(o => o._regionId && (o.type === 'i-text' || o.type === 'textbox'));
    }
    if (active._regionId && (active.type === 'i-text' || active.type === 'textbox')) return [active];
    return [];
  },

  // ── Font Search (toolbar) ──
  onFontSearch(query) {
    clearTimeout(this._fontSearchTimer);
    this._fontSearchTimer = setTimeout(() => this._renderFontDropdown(query), 100);
  },

  onFontSearchFocus() {
    if (document.getElementById('font-dropdown').children.length === 0) {
      this._renderFontDropdown('');
    }
    document.getElementById('font-dropdown').style.display = 'block';
  },

  onFontSearchBlur() {
    setTimeout(() => {
      document.getElementById('font-dropdown').style.display = 'none';
    }, 200);
  },

  // ── Font Search (context menu) ──
  onCtxFontSearch(query) {
    clearTimeout(this._fontSearchTimer);
    this._fontSearchTimer = setTimeout(() => this._renderFontDropdownCtx(query), 100);
  },

  onCtxFontSearchFocus() {
    if (document.getElementById('ctx-font-dropdown').children.length === 0) {
      this._renderFontDropdownCtx('');
    }
    document.getElementById('ctx-font-dropdown').style.display = 'block';
  },

  onCtxFontSearchBlur() {
    setTimeout(() => {
      document.getElementById('ctx-font-dropdown').style.display = 'none';
    }, 200);
  },

  _renderFontDropdown(query, dropdownId = 'font-dropdown', inputId = 'font-family-input') {
    const dropdown = document.getElementById(dropdownId);
    const q = query.toLowerCase().trim();
    let matches = this.allFonts;
    if (q) {
      matches = this.allFonts.filter(f => f.toLowerCase().includes(q));
    }
    const show = matches.slice(0, 100);

    const targets = this._getTextTargets();
    const current = targets.length > 0 ? (targets[0].fontFamily || this.currentFont) : this.currentFont;

    dropdown.innerHTML = show.map(f =>
      `<div class="font-dropdown-item" style="font-family:'${f.replace(/'/g, "\\'")}'" onmousedown="TextboxManager.selectFont('${f.replace(/'/g, "\\'")}', '${dropdownId}', '${inputId}')">${f}</div>`
    ).join('');

    if (show.length === 0) {
      dropdown.innerHTML = '<div class="font-dropdown-empty">未找到匹配字体</div>';
    }

    // Add priority section label if showing unfiltered
    if (!q) {
      const firstPriority = show.find(f => this.PRIORITY_FONTS.some(p => p.toLowerCase() === f.toLowerCase()));
      if (firstPriority) {
        // Add a label before first match
      }
    }

    dropdown.querySelectorAll('.font-dropdown-item').forEach(el => {
      const fn = el.style.fontFamily.replace(/'/g, '');
      el.classList.toggle('active', fn.toLowerCase() === current.toLowerCase());
    });

    dropdown.style.display = 'block';
  },

  _renderFontDropdownCtx(query) {
    this._renderFontDropdown(query, 'ctx-font-dropdown', 'ctx-font-input');
  },

  selectFont(fontName, dropdownId, inputId) {
    this.currentFont = fontName;
    document.getElementById(inputId || 'font-family-input').value = fontName;
    document.getElementById(dropdownId || 'font-dropdown').style.display = 'none';
    const targets = this._getTextTargets();
    if (targets.length > 0) {
      for (const obj of targets) {
        obj.set('fontFamily', fontName);
      }
      this.canvas().renderAll();
      this.onObjectModified({ target: targets[0] });
    }
  },

  // ── Text Objects ──
  _createOneTextObject(r) {
    if (!r.bbox || r.bbox.length < 4) return null;
    const [x1, y1, x2, y2] = r.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    if (w < 5 || h < 5) return null;

    const isFormula = r.is_formula === true;
    const latex = r.latex_source || '';

    // Formula region: render as KaTeX image instead of IText
    if (isFormula && latex) {
      return { _pendingFormula: true, region: r, pos: { x1, y1, x2, y2 } };
    }

    const text = r.final_text || r.ocr?.best_text || '';
    const style = r.style || {};
    const accentColor = isFormula ? '#00d2a0' : '#6c5ce7';

    const itext = new fabric.IText(text, {
      left: x1, top: y1,
      width: Math.max(w, 30),
      fontSize: style.font_size || Math.max(h * 0.7, 14),
      fontFamily: style.font_family || 'Microsoft YaHei',
      fill: style.color ? `rgba(${style.color.join(',')})` : '#000000',
      fontWeight: style.font_weight || 'normal',
      fontStyle: style.italic ? 'italic' : 'normal',
      underline: style.underline || false,
      linethrough: style.strikethrough || false,
      textAlign: style.align || 'left',
      _regionId: r.id,
      _isFormula: false,
      _latexSource: '',
      editable: true,
      lockScalingFlip: true,
      cornerSize: 8,
      cornerColor: accentColor,
      cornerStrokeColor: '#fff',
      transparentCorners: false,
      borderColor: accentColor,
      selectionBackgroundColor: isFormula ? 'rgba(0,210,160,0.15)' : 'rgba(108,92,231,0.15)',
    });

    itext.setControlsVisibility({ mt: false, mb: false, ml: false, mr: false, mtr: true });
    return itext;
  },

  // Render a formula region as a KaTeX Fabric.Image
  _renderFormulaImage(r, callback) {
    const latex = r.latex_source || r.final_text || '';
    if (!latex) return;

    const [x1, y1, x2, y2] = r.bbox;
    const w = Math.max(x2 - x1, 30);

    try {
      let katexCss = '';
      try {
        for (const sheet of document.styleSheets) {
          if (sheet.href && sheet.href.includes('katex')) {
            for (const rule of sheet.cssRules) {
              katexCss += rule.cssText + '\n';
            }
          }
        }
      } catch (e) { /* cross-origin stylesheet */ }

      const katexHtml = katex.renderToString(latex, {
        throwOnError: false, displayMode: false, trust: true,
      });

      const measure = document.createElement('div');
      measure.style.cssText = 'position:fixed;visibility:hidden;display:inline-block;';
      if (katexCss) {
        const style = document.createElement('style');
        style.textContent = katexCss;
        measure.appendChild(style);
      }
      measure.insertAdjacentHTML('beforeend', katexHtml);
      document.body.appendChild(measure);
      const imgW = Math.max(Math.ceil(measure.offsetWidth) + 8, 40);
      const imgH = Math.max(Math.ceil(measure.offsetHeight) + 4, 20);
      document.body.removeChild(measure);

      const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="${imgW}" height="${imgH}">
<defs><style>${katexCss}</style></defs>
<foreignObject width="100%" height="100%">
<div xmlns="http://www.w3.org/1999/xhtml">${katexHtml}</div>
</foreignObject>
</svg>`;
      const svgBase64 = btoa(unescape(encodeURIComponent(svgContent)));
      const dataUrl = 'data:image/svg+xml;base64,' + svgBase64;

      fabric.Image.fromURL(dataUrl, (img) => {
        img.set({
          left: x1, top: y1,
          _isFormula: true,
          _latexSource: latex,
          _regionId: r.id,
          selectable: true,
          hasControls: true,
          hasBorders: true,
          cornerSize: 8,
          cornerColor: '#00d2a0',
          cornerStrokeColor: '#fff',
          borderColor: '#00d2a0',
          transparentCorners: false,
        });
        if (callback) callback(img);
      }, { crossOrigin: 'anonymous' });
    } catch (e) {
      // Fallback: create IText with the LaTeX source as text
      console.warn('Formula render failed, using text fallback:', e);
      const itext = new fabric.IText(latex.substring(0, 100), {
        left: x1, top: y1,
        width: Math.max(w, 30),
        fontSize: 16,
        fontFamily: 'monospace',
        fill: '#888888',
        _regionId: r.id,
        _isFormula: true,
        _latexSource: latex,
        editable: false,
        cornerSize: 8,
        cornerColor: '#00d2a0',
        cornerStrokeColor: '#fff',
        transparentCorners: false,
        borderColor: '#00d2a0',
      });
      itext.setControlsVisibility({ mt: false, mb: false, ml: false, mr: false, mtr: true });
      if (callback) callback(itext);
    }
  },

  createTextObjects(regions) {
    this.regions = regions;
    this.textObjects = {};
    const pendingFormulas = [];

    for (const r of regions) {
      const result = this._createOneTextObject(r);
      if (!result) continue;
      if (result._pendingFormula) {
        pendingFormulas.push(result);
      } else {
        this.canvas().add(result);
        this.textObjects[r.id] = result;
      }
    }

    this.canvas().renderAll();
    this.updateRegionList();

    // Render formulas asynchronously (KaTeX → SVG → Fabric Image)
    let pendingCount = pendingFormulas.length;
    if (pendingCount === 0) {
      if (typeof UndoManager !== 'undefined') UndoManager.saveState('创建文字框');
      return;
    }

    for (const pf of pendingFormulas) {
      this._renderFormulaImage(pf.region, (img) => {
        this.canvas().add(img);
        this.textObjects[pf.region.id] = img;
        pendingCount--;
        if (pendingCount === 0) {
          this.canvas().renderAll();
          this.updateRegionList();
          if (typeof UndoManager !== 'undefined') UndoManager.saveState('创建文字框');
        }
      });
    }
  },

  canvas() { return CanvasManager.canvas; },

  onSelectionChanged(e) {
    const active = this.canvas().getActiveObject();
    if (!active) {
      document.getElementById('font-controls').style.display = 'none';
      return;
    }
    // Check if any of the selected objects is a text box
    const targets = this._getTextTargets();
    if (targets.length > 0) {
      document.getElementById('font-controls').style.display = 'flex';
      this._syncUI(targets[0]);
      if (targets[0]._regionId) this._highlightRegionItem(targets[0]._regionId);
    } else {
      document.getElementById('font-controls').style.display = 'none';
    }
  },

  onSelectionCleared() {
    document.getElementById('font-controls').style.display = 'none';
    this._highlightRegionItem(null);
  },

  _parseColor(css) {
    if (!css || css === 'transparent') return [0, 0, 0, 255];
    if (Array.isArray(css)) return css;
    if (css.startsWith('rgba')) {
      const m = css.match(/[\d.]+/g);
      if (m) return [parseInt(m[0]), parseInt(m[1]), parseInt(m[2]), Math.round(parseFloat(m[3]||1)*255)];
    }
    if (css.startsWith('rgb')) {
      const m = css.match(/\d+/g);
      if (m) return [parseInt(m[0]), parseInt(m[1]), parseInt(m[2]), 255];
    }
    if (css.startsWith('#')) {
      const h = css.slice(1);
      if (h.length === 3) return [parseInt(h[0]+h[0],16), parseInt(h[1]+h[1],16), parseInt(h[2]+h[2],16), 255];
      if (h.length === 6) return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16), 255];
      if (h.length === 8) return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16), parseInt(h.slice(6,8),16)];
    }
    return [0, 0, 0, 255];
  },

  onObjectModified(e) {
    const obj = e.target;
    if (!obj || !obj._regionId) return;
    const rid = obj._regionId;

    const colorTuple = this._parseColor(obj.fill);
    API.updateRegion(App.currentProjectId, rid, {
      final_text: obj.text,
      style: {
        font_family: obj.fontFamily,
        font_size: obj.fontSize,
        font_weight: obj.fontWeight,
        italic: obj.fontStyle === 'italic',
        underline: obj.underline,
        strikethrough: obj.linethrough || false,
        align: obj.textAlign,
        color: colorTuple,
      },
    }).catch(e => console.warn('Sync failed:', e));

    if (typeof UndoManager !== 'undefined') UndoManager.saveState('修改文字/样式');
  },

  _syncUI(obj) {
    const fontInput = document.getElementById('font-family-input');
    fontInput.value = obj.fontFamily || 'Arial';
    this.currentFont = obj.fontFamily || 'Arial';

    document.getElementById('font-size').value = obj.fontSize || 24;
    document.getElementById('text-color').value = obj.fill || '#000000';
    this.currentBold = obj.fontWeight === 'bold';
    this.currentItalic = obj.fontStyle === 'italic';
    this.currentUnderline = obj.underline || false;
    this.currentStrikethrough = obj.linethrough || false;

    document.getElementById('btn-bold').classList.toggle('active', this.currentBold);
    document.getElementById('btn-italic').classList.toggle('active', this.currentItalic);
    document.getElementById('btn-underline').classList.toggle('active', this.currentUnderline);
  },

  // ── Style Application (handles multi-select) ──
  applyStyle(prop, value) {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    for (const obj of targets) {
      obj.set(prop, value);
    }
    this.canvas().renderAll();
    this._syncUI(targets[0]);
    this.onObjectModified({ target: targets[0] });
  },

  toggleBold() {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    this.currentBold = !this.currentBold;
    for (const obj of targets) {
      obj.set('fontWeight', this.currentBold ? 'bold' : 'normal');
    }
    this.canvas().renderAll();
    this._syncUI(targets[0]);
    this.onObjectModified({ target: targets[0] });
  },

  toggleItalic() {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    this.currentItalic = !this.currentItalic;
    for (const obj of targets) {
      obj.set('fontStyle', this.currentItalic ? 'italic' : 'normal');
    }
    this.canvas().renderAll();
    this._syncUI(targets[0]);
    this.onObjectModified({ target: targets[0] });
  },

  toggleUnderline() {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    this.currentUnderline = !this.currentUnderline;
    for (const obj of targets) {
      obj.set('underline', this.currentUnderline);
    }
    this.canvas().renderAll();
    this._syncUI(targets[0]);
    this.onObjectModified({ target: targets[0] });
  },

  toggleStrikethrough() {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    this.currentStrikethrough = !this.currentStrikethrough;
    for (const obj of targets) {
      obj.set('linethrough', this.currentStrikethrough);
    }
    this.canvas().renderAll();
    this._syncUI(targets[0]);
    this.onObjectModified({ target: targets[0] });
  },

  applyAlign(align) {
    const targets = this._getTextTargets();
    if (targets.length === 0) return;
    for (const obj of targets) {
      obj.set('textAlign', align);
    }
    this.canvas().renderAll();
    this.onObjectModified({ target: targets[0] });
  },

  deleteSelected() {
    const targets = this._getTextTargets();
    if (targets.length === 0) {
      // also handle non-text objects (color blocks, formula images)
      const active = this.canvas().getActiveObject();
      if (active && (active._boxType || active._isFormula)) {
        const rid = active._regionId;
        this.canvas().remove(active);
        if (rid) {
          delete this.textObjects[rid];
          this.regions = this.regions.filter(r => r.id !== rid);
        }
        this.canvas().renderAll();
        this.updateRegionList();
        if (typeof UndoManager !== 'undefined') UndoManager.saveState('删除对象');
        return;
      }
      return;
    }
    const count = targets.length;
    for (const obj of targets) {
      const rid = obj._regionId;
      this.canvas().remove(obj);
      delete this.textObjects[rid];
      this.regions = this.regions.filter(r => r.id !== rid);
    }
    this.canvas().discardActiveObject();
    this.canvas().renderAll();
    this.updateRegionList();
    document.getElementById('font-controls').style.display = 'none';
    App.toast(`已删除 ${count} 个文字框`, 'info');
    if (typeof UndoManager !== 'undefined') UndoManager.saveState('删除文字框');
  },

  selectByRegionId(rid) {
    const obj = this.textObjects[rid];
    if (obj) {
      this.canvas().setActiveObject(obj);
      this.canvas().renderAll();
      this._highlightRegionItem(rid);
    }
  },

  selectAllInRect(rect) {
    const selected = [];
    for (const [rid, obj] of Object.entries(this.textObjects)) {
      const objW = (obj.width || (obj.getScaledWidth && obj.getScaledWidth()) || 100) * (obj.scaleX || 1);
      const objH = (obj.height || (obj.getScaledHeight && obj.getScaledHeight()) || 24) * (obj.scaleY || 1);
      const cx = obj.left + objW / 2;
      const cy = obj.top + objH / 2;
      if (cx >= rect.left && cx <= rect.left + rect.width &&
          cy >= rect.top && cy <= rect.top + rect.height) {
        selected.push(obj);
      }
    }
    if (selected.length > 0) {
      const sel = new fabric.ActiveSelection(selected, { canvas: this.canvas() });
      this.canvas().setActiveObject(sel);
      this.canvas().renderAll();
      document.getElementById('font-controls').style.display = 'flex';
    }
    return selected.length;
  },

  // ── New blank text box ──
  createNewTextBox() {
    const canvas = this.canvas();
    const vpt = canvas.viewportTransform;
    const centerX = (canvas.getWidth() / 2 - (vpt[4] || 0)) / (vpt[0] || 1);
    const centerY = (canvas.getHeight() / 2 - (vpt[5] || 0)) / (vpt[3] || 1);

    const newId = 'textbox_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
    const itext = new fabric.IText('新文字', {
      left: centerX - 60,
      top: centerY - 16,
      width: 200,
      fontSize: 24,
      fontFamily: this.currentFont || 'Microsoft YaHei',
      fill: '#000000',
      fontWeight: 'normal',
      fontStyle: 'normal',
      underline: false,
      linethrough: false,
      textAlign: 'left',
      _regionId: newId,
      editable: true,
      lockScalingFlip: true,
      cornerSize: 8,
      cornerColor: '#6c5ce7',
      cornerStrokeColor: '#fff',
      transparentCorners: false,
      borderColor: '#6c5ce7',
      selectionBackgroundColor: 'rgba(108,92,231,0.15)',
    });

    itext.setControlsVisibility({ mt: false, mb: false, ml: false, mr: false, mtr: true });

    canvas.add(itext);
    canvas.setActiveObject(itext);
    canvas.renderAll();

    this.textObjects[newId] = itext;
    this.regions.push({
      id: newId,
      bbox: [itext.left, itext.top, itext.left + 200, itext.top + 28],
      final_text: '新文字',
      style: {
        font_family: 'Microsoft YaHei',
        font_size: 24,
        font_weight: 'normal',
        italic: false,
        underline: false,
        strikethrough: false,
        align: 'left',
        color: [0, 0, 0, 255],
      },
    });
    this.updateRegionList();
    document.getElementById('font-controls').style.display = 'flex';
    this._syncUI(itext);
    App.toast('已创建新文字框', 'success');
    if (typeof UndoManager !== 'undefined') UndoManager.saveState('新建文字框');
  },

  selectAllTextboxes() {
    const all = Object.values(this.textObjects);
    if (all.length > 0) {
      const sel = new fabric.ActiveSelection(all, { canvas: this.canvas() });
      this.canvas().setActiveObject(sel);
      this.canvas().renderAll();
      document.getElementById('font-controls').style.display = 'flex';
      App.toast(`已选择 ${all.length} 个文字框`, 'info');
    }
  },

  deselectAll() {
    this.canvas().discardActiveObject();
    this.canvas().renderAll();
    document.getElementById('font-controls').style.display = 'none';
    this._highlightRegionItem(null);
  },

  _highlightRegionItem(rid) {
    document.querySelectorAll('.region-item').forEach(el => el.classList.remove('active'));
    if (rid) {
      const el = document.querySelector(`.region-item[data-rid="${rid}"]`);
      if (el) el.classList.add('active');
    }
  },

  filterRegions(query) {
    const q = query.toLowerCase();
    document.querySelectorAll('.region-item').forEach(el => {
      const text = (el.getAttribute('data-text') || '').toLowerCase();
      const id = (el.getAttribute('data-rid') || '').toLowerCase();
      el.style.display = (!q || text.includes(q) || id.includes(q)) ? '' : 'none';
    });
  },

  updateRegionList() {
    const list = document.getElementById('region-list');
    const count = document.getElementById('region-count');
    count.textContent = this.regions.length;

    const FLAG_LABELS = {
      number: '数字', currency: '金额', percent: '百分比', date: '日期',
      time: '时间', phone_cn: '电话', url: '链接', email: '邮箱', model: '型号',
    };

    const CRITICAL_FLAGS = new Set(['phone_cn', 'email', 'currency', 'url']);

    list.innerHTML = this.regions.map(r => {
      const isFormula = r.is_formula === true;
      const rawText = r.final_text || r.ocr?.best_text || '';
      const displayText = (isFormula ? '[公式] ' : '') + (rawText || '(空)').substring(0, 18);
      const idSuffix = r.id.slice(-6);
      const flags = r.risk_flags || [];
      const badgesHtml = flags.length > 0
        ? `<div class="risk-badges">${flags.map(f => {
            const isCritical = CRITICAL_FLAGS.has(f);
            return `<span class="risk-badge risk-${f}${isCritical ? ' risk-critical' : ''}">${FLAG_LABELS[f] || f}</span>`;
          }).join('')}</div>` : '';
      const formulaClass = isFormula ? ' region-item-formula' : '';
      return `<div class="region-item${formulaClass}" data-rid="${r.id}" data-text="${displayText}" onclick="TextboxManager.selectByRegionId('${r.id}')">
        <span class="rid">${idSuffix}</span>
        <span class="rtext">${displayText}</span>
        ${badgesHtml}
      </div>`;
    }).join('');
  },

  addRegions(newRegions) {
    let added = 0;
    const pendingFormulas = [];
    for (const r of newRegions) {
      if (this.regions.find(ex => ex.id === r.id)) continue;
      this.regions.push(r);
      const result = this._createOneTextObject(r);
      if (!result) continue;
      if (result._pendingFormula) {
        pendingFormulas.push(result);
      } else {
        this.canvas().add(result);
        this.textObjects[r.id] = result;
        added++;
      }
    }
    if (added > 0 || pendingFormulas.length > 0) {
      this.canvas().renderAll();
      this.updateRegionList();
    }
    let pendingCount = pendingFormulas.length;
    for (const pf of pendingFormulas) {
      this._renderFormulaImage(pf.region, (img) => {
        this.canvas().add(img);
        this.textObjects[pf.region.id] = img;
        pendingCount--;
        if (pendingCount === 0) {
          this.canvas().renderAll();
          this.updateRegionList();
        }
      });
    }
  },

  // ── Rebuild text objects map (for undo/redo) ──
  _rebuildTextObjectsMap() {
    this.textObjects = {};
    const objects = this.canvas().getObjects();
    for (const obj of objects) {
      if (obj._regionId && (obj.type === 'i-text' || obj._isFormula)) {
        this.textObjects[obj._regionId] = obj;
      }
    }
    this.regions = Object.values(this.textObjects).map(obj => ({
      id: obj._regionId,
      bbox: [obj.left, obj.top, obj.left + (obj.width||100)*(obj.scaleX||1), obj.top + (obj.height||24)*(obj.scaleY||1)],
      final_text: obj._latexSource || obj.text || '',
      is_formula: obj._isFormula || false,
      latex_source: obj._latexSource || '',
      style: {
        font_family: obj.fontFamily || '',
        font_size: obj.fontSize || 16,
        font_weight: obj.fontWeight || 'normal',
        italic: obj.fontStyle === 'italic' || false,
        underline: obj.underline || false,
        strikethrough: obj.linethrough || false,
        align: obj.textAlign || 'left',
        color: this._parseColor(obj.fill || '#000000'),
      },
    }));
    this.updateRegionList();
  },

  batchApplyStyle(style) {
    const ids = Object.keys(this.textObjects);
    for (const rid of ids) {
      const obj = this.textObjects[rid];
      if (style.fontFamily) obj.set('fontFamily', style.fontFamily);
      if (style.fontSize) obj.set('fontSize', style.fontSize);
      if (style.fontWeight) obj.set('fontWeight', style.fontWeight);
      if (style.fill) obj.set('fill', style.fill);
    }
    this.canvas().renderAll();

    API.batchUpdateRegions(App.currentProjectId, ids, {
      font_family: style.fontFamily,
      font_size: style.fontSize,
      font_weight: style.fontWeight,
    }).catch(e => console.warn('Batch sync failed:', e));

    if (typeof UndoManager !== 'undefined') UndoManager.saveState('批量样式');
  },
};
