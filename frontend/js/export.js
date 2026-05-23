const Export = {
  init() {
    const q = document.getElementById('export-quality');
    q.addEventListener('input', () => {
      document.getElementById('export-quality-val').textContent = q.value;
    });
  },

  open() {
    document.getElementById('overlay-export').style.display = 'flex';
  },

  close() {
    document.getElementById('overlay-export').style.display = 'none';
  },

  async doExport() {
    const format = document.getElementById('export-format').value;
    const scale = parseInt(document.getElementById('export-scale').value);
    const quality = parseInt(document.getElementById('export-quality').value);

    App.showLoading('正在导出...');
    try {
      const canvas = CanvasManager.canvas;
      const bgImg = CanvasManager.cleanImg || CanvasManager.originalImg;
      if (!bgImg) throw new Error('没有可导出的图像');

      const targetW = bgImg.width * scale;
      const targetH = bgImg.height * scale;

      const offCanvas = document.createElement('canvas');
      offCanvas.width = targetW;
      offCanvas.height = targetH;
      const ctx = offCanvas.getContext('2d');

      // Draw background
      const bgEl = bgImg.getElement();
      ctx.drawImage(bgEl, 0, 0, targetW, targetH);

      // Draw text objects with proper scaling
      const textObjs = canvas.getObjects().filter(o => o._regionId && o.text);
      for (const obj of textObjs) {
        ctx.save();
        const x = obj.left * scale;
        const y = obj.top * scale;
        const fs = obj.fontSize * scale;
        const fw = (obj.width * obj.scaleX) * scale;

        const weight = obj.fontWeight || 'normal';
        const style = obj.fontStyle || 'normal';
        ctx.font = `${style} ${weight} ${fs}px ${obj.fontFamily || 'Arial'}`;
        ctx.fillStyle = obj.fill || '#000000';
        ctx.textAlign = obj.textAlign || 'left';
        ctx.textBaseline = 'top';

        const lines = (obj.text || '').split('\n');
        const lineHeight = fs * 1.2;
        let tx = x;
        if (obj.textAlign === 'center') tx = x + fw / 2;
        else if (obj.textAlign === 'right') tx = x + fw;

        for (let i = 0; i < lines.length; i++) {
          const ly = y + i * lineHeight;
          ctx.fillText(lines[i], tx, ly);
          if (obj.underline) {
            const metrics = ctx.measureText(lines[i]);
            let ux = tx;
            if (obj.textAlign === 'center') ux = tx - metrics.width / 2;
            else if (obj.textAlign === 'right') ux = tx - metrics.width;
            ctx.fillRect(ux, ly + fs + 1, metrics.width, Math.max(1, fs * 0.08));
          }
        }
        ctx.restore();
      }

      // Draw formula images
      const formulaObjs = canvas.getObjects().filter(o => o._isFormula);
      for (const obj of formulaObjs) {
        ctx.save();
        const imgEl = obj.getElement();
        if (imgEl) {
          ctx.drawImage(imgEl, obj.left * scale, obj.top * scale,
            obj.width * obj.scaleX * scale, obj.height * obj.scaleY * scale);
        }
        ctx.restore();
      }

      // Draw color block overlays
      const blockObjs = canvas.getObjects().filter(o => o._boxType === 'color-block');
      for (const obj of blockObjs) {
        ctx.save();
        ctx.globalAlpha = obj.opacity || 0.85;
        ctx.fillStyle = obj.fill;
        ctx.fillRect(obj.left * scale, obj.top * scale,
          obj.width * obj.scaleX * scale, obj.height * obj.scaleY * scale);
        ctx.restore();
      }

      const mime = format === 'jpeg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png';
      offCanvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `textpatch_${Date.now()}.${format === 'jpeg' ? 'jpg' : format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        App.setStatus(`导出完成 — ${targetW}×${targetH} ${format.toUpperCase()}`);
        App.hideLoading();
        App.toast(`导出完成 — ${targetW}×${targetH} ${format.toUpperCase()}`, 'success');
      }, mime, quality / 100);
    } catch (e) {
      App.hideLoading();
      App.toast('导出失败: ' + e.message, 'error');
    }
    this.close();
  },
};
