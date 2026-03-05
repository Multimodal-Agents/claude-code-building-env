'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Apply electron-mode class to <body> as early as possible.
// This runs BEFORE any page JS, so CSS that depends on the class
// (rz resize handles, win-ctrl buttons, etc.) works immediately.
function applyElectronClass() {
  if (document.body) {
    document.body.classList.add('electron-mode');
  } else {
    document.addEventListener('DOMContentLoaded', () => {
      document.body.classList.add('electron-mode');
    }, { once: true, capture: true });
  }
}
applyElectronClass();

/**
 * Expose a safe subset of Electron APIs to the renderer (index.html).
 * The renderer can call window.codeGraph.* to trigger native functionality.
 */
contextBridge.exposeInMainWorld('codeGraph', {
  /** Open a native folder picker and register the selected project. */
  selectFolder: () => ipcRenderer.invoke('dialog:openDirectory'),

  /** Return the list of recently opened project paths. */
  getRecentProjects: () => ipcRenderer.invoke('store:getRecentProjects'),

  /** Return the last-used project path. */
  getLastProject: () => ipcRenderer.invoke('store:getLastProject'),

  /** Frameless window controls */
  window: {
    minimize:      () => ipcRenderer.send('win:minimize'),
    maximize:      () => ipcRenderer.send('win:maximize'),
    close:         () => ipcRenderer.send('win:close'),
    /** cb({ maximized: bool }) — called whenever the window is maximized/restored */
    onStateChange: (cb) => ipcRenderer.on('win:state', (_evt, data) => cb(data)),
    /** JS-driven window drag (replaces unreliable -webkit-app-region on localhost) */
    startDrag:     ()    => ipcRenderer.send('win:drag-start'),
    endDrag:       ()    => ipcRenderer.send('win:drag-end'),
    /** Custom resize (replaces OS resize handles lost with frame:false on Windows) */
    startResize:   (dir) => ipcRenderer.send('win:resize-start', dir),
    endResize:     ()    => ipcRenderer.send('win:resize-end'),
  },

  /** True when running inside Electron (lets the UI adapt). */
  isElectron: true,
});
