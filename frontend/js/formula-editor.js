const FormulaEditor = {
  init() {
    const input = document.getElementById('formula-input');
    input.addEventListener('input', () => this.updatePreview());
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') { e.preventDefault(); this._insertText('  '); }
    });
    this._buildSymbolPalette();
  },

  open() {
    document.getElementById('formula-input').value = '';
    document.getElementById('overlay-formula').style.display = 'flex';
    document.getElementById('formula-input').focus();
    this.updatePreview();
  },

  close() {
    document.getElementById('overlay-formula').style.display = 'none';
  },

  updatePreview() {
    const input = document.getElementById('formula-input').value;
    const preview = document.getElementById('formula-preview');
    if (!input.trim()) {
      preview.innerHTML = '<span class="formula-placeholder">在上方输入LaTeX公式查看预览</span>';
      return;
    }
    try {
      katex.render(input, preview, { throwOnError: false, displayMode: true, trust: true });
    } catch (e) {
      preview.innerHTML = `<span class="formula-error">渲染错误: ${e.message}</span>`;
    }
  },

  _getInsertPosition() {
    // If a text box is selected, place the formula at its position
    const active = CanvasManager.canvas.getActiveObject();
    if (active && active._regionId && (active.type === 'i-text' || active.type === 'textbox')) {
      return { left: active.left, top: active.top + (active.height || 0) * (active.scaleY || 1) + 8 };
    }
    // Otherwise, place at viewport center
    const canvas = CanvasManager.canvas;
    const vpt = canvas.viewportTransform;
    return {
      left: (canvas.getWidth() / 2 - (vpt[4] || 0)) / (vpt[0] || 1),
      top: (canvas.getHeight() / 2 - (vpt[5] || 0)) / (vpt[3] || 1),
    };
  },

  insert() {
    const input = document.getElementById('formula-input').value.trim();
    if (!input) return;

    const insertPos = this._getInsertPosition();

    try {
      // Extract KaTeX CSS from loaded stylesheet for inline embedding
      let katexCss = '';
      try {
        for (const sheet of document.styleSheets) {
          if (sheet.href && sheet.href.includes('katex')) {
            for (const rule of sheet.cssRules) {
              katexCss += rule.cssText + '\n';
            }
          }
        }
      } catch (e) { /* cross-origin stylesheet, continue without inline CSS */ }

      // Render KaTeX to HTML string
      const katexHtml = katex.renderToString(input, {
        throwOnError: false, displayMode: false, trust: true,
      });

      // Measure rendered dimensions using a hidden temp div
      const measure = document.createElement('div');
      measure.style.cssText = 'position:fixed;visibility:hidden;display:inline-block;';
      if (katexCss) {
        const style = document.createElement('style');
        style.textContent = katexCss;
        measure.appendChild(style);
      }
      measure.insertAdjacentHTML('beforeend', katexHtml);
      document.body.appendChild(measure);
      const w = Math.max(Math.ceil(measure.offsetWidth) + 8, 40);
      const h = Math.max(Math.ceil(measure.offsetHeight) + 4, 20);
      document.body.removeChild(measure);

      // Build SVG with inline CSS and foreignObject for reliable rendering
      const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
<defs><style>${katexCss}</style></defs>
<foreignObject width="100%" height="100%">
<div xmlns="http://www.w3.org/1999/xhtml">${katexHtml}</div>
</foreignObject>
</svg>`;
      const svgBase64 = btoa(unescape(encodeURIComponent(svgContent)));
      const dataUrl = 'data:image/svg+xml;base64,' + svgBase64;

      const canvas = CanvasManager.canvas;
      fabric.Image.fromURL(dataUrl, (img) => {
        img.set({
          left: insertPos.left,
          top: insertPos.top,
          _isFormula: true,
          _latexSource: input,
          _regionId: 'formula_' + Date.now(),
          selectable: true,
          hasControls: true,
          hasBorders: true,
          cornerSize: 8,
          cornerColor: '#00d2a0',
          cornerStrokeColor: '#fff',
          borderColor: '#00d2a0',
          transparentCorners: false,
        });

        canvas.add(img);
        canvas.setActiveObject(img);
        canvas.renderAll();
        TextboxManager.updateRegionList();
        App.toast('公式已插入', 'success');
        if (typeof UndoManager !== 'undefined') UndoManager.saveState('插入公式');
      }, { crossOrigin: 'anonymous' });
    } catch (e) {
      // Fallback: render as plain text on canvas
      const c = document.createElement('canvas');
      c.width = Math.max(input.length * 14 + 16, 40);
      c.height = 30;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.font = '16px serif';
      ctx.fillText(input.substring(0, 50), 8, 22);
      const fallbackUrl = c.toDataURL();

      const canvas = CanvasManager.canvas;
      fabric.Image.fromURL(fallbackUrl, (img) => {
        img.set({
          left: insertPos.left,
          top: insertPos.top,
          _isFormula: true, _latexSource: input,
          _regionId: 'formula_' + Date.now(),
          selectable: true, hasControls: true, hasBorders: true,
          cornerSize: 8, cornerColor: '#00d2a0', cornerStrokeColor: '#fff',
          borderColor: '#00d2a0', transparentCorners: false,
        });
        canvas.add(img);
        canvas.setActiveObject(img);
        canvas.renderAll();
        TextboxManager.updateRegionList();
        App.toast('公式已插入（文本模式）', 'warning');
        if (typeof UndoManager !== 'undefined') UndoManager.saveState('插入公式');
      }, { crossOrigin: 'anonymous' });
    }

    this.close();
  },

  _insertText(text) {
    const input = document.getElementById('formula-input');
    const start = input.selectionStart;
    const end = input.selectionEnd;
    input.value = input.value.slice(0, start) + text + input.value.slice(end);
    input.focus();
    input.setSelectionRange(start + text.length, start + text.length);
    this.updatePreview();
  },

  _buildSymbolPalette() {
    const palette = document.getElementById('formula-symbol-palette');
    const groups = [
      { name: '希腊', symbols: ['\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\theta', '\\lambda', '\\mu', '\\pi', '\\sigma', '\\phi', '\\omega', '\\Gamma', '\\Delta', '\\Theta', '\\Sigma', '\\Omega'] },
      { name: '运算', symbols: ['+', '-', '\\times', '\\cdot', '\\div', '\\pm', '\\mp', '\\leq', '\\geq', '\\neq', '\\approx', '\\equiv', '\\infty'] },
      { name: '结构', symbols: ['\\frac{}{}', '\\sqrt{}', '\\sqrt[n]{}', '\\sum_{}^{}', '\\int_{}^{}', '\\prod_{}^{}', '\\lim_{x \\to \\infty}', '^{}', '_{}', '\\hat{}', '\\bar{}', '\\vec{}', '\\overline{}', '\\underline{}'] },
      { name: '括号', symbols: ['(', ')', '[', ']', '\\{', '\\}', '\\langle', '\\rangle', '\\left(', '\\right)', '\\left[', '\\right]'] },
      { name: '函数', symbols: ['\\sin', '\\cos', '\\tan', '\\log', '\\ln', '\\exp', '\\max', '\\min', '\\det', '\\gcd', '\\operatorname{}'] },
    ];

    palette.innerHTML = groups.map(g => {
      const btns = g.symbols.map(s => {
        const safeLabel = s.replace(/[<>&"']/g, '');
        const display = s.replace(/_\{.*?\}/g, '↓').replace(/\^\{.*?\}/g, '↑').replace(/\\/g, '');
        return `<button class="sym-btn" title="${safeLabel}" onclick="FormulaEditor._insertText('${s.replace(/'/g, "\\'")}')">${display.substring(0, 6)}</button>`;
      }).join('');
      return `<div class="sym-group"><span class="sym-group-name">${g.name}</span><div class="sym-row">${btns}</div></div>`;
    }).join('');
  },
};
