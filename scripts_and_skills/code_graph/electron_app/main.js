'use strict';

const { app, BrowserWindow, dialog, ipcMain, Menu, shell, screen } = require('electron');
const { spawn } = require('child_process');
const path  = require('path');
const fs    = require('fs');
const http  = require('http');

// Guard against EPIPE when launched without a console (desktop shortcut / hidden window)
process.stdout.on('error', () => {});
process.stderr.on('error', () => {});
process.on('uncaughtException', err => {
  if (err.code === 'EPIPE') return;  // swallow broken-pipe — no console to write to
  // Re-throw non-EPIPE errors so Electron's default handler can show the dialog
  throw err;
});

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT       = parseInt(process.env.CGRAPH_PORT || '8765', 10);
const SERVER_URL = `http://localhost:${PORT}`;
const REPO_ROOT  = path.resolve(__dirname, '..', '..', '..');  // claude_code_building_env/

// ---------------------------------------------------------------------------
// Simple JSON store (persists last-used projects)
// ---------------------------------------------------------------------------
const STORE_PATH = path.join(app.getPath('userData'), 'code-graph-store.json');

function loadStore() {
  try { return JSON.parse(fs.readFileSync(STORE_PATH, 'utf8')); }
  catch { return { lastProject: null, recentProjects: [] }; }
}

function saveStore(data) {
  fs.mkdirSync(path.dirname(STORE_PATH), { recursive: true });
  fs.writeFileSync(STORE_PATH, JSON.stringify(data, null, 2));
}

function addRecentProject(projectPath) {
  const store = loadStore();
  store.lastProject = projectPath;
  store.recentProjects = [
    projectPath,
    ...store.recentProjects.filter(p => p !== projectPath),
  ].slice(0, 10);
  saveStore(store);
}

// ---------------------------------------------------------------------------
// Python detection
// ---------------------------------------------------------------------------
function findPython() {
  const candidates = [
    path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(REPO_ROOT, 'venv',  'Scripts', 'python.exe'),
    path.join(REPO_ROOT, '.venv', 'bin', 'python'),
    path.join(REPO_ROOT, 'venv',  'bin', 'python'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return 'python';  // fall back to PATH
}

// ---------------------------------------------------------------------------
// Server management
// ---------------------------------------------------------------------------
let serverProcess = null;

function startServer(projectPath) {
  const python = findPython();
  const args   = ['-m', 'scripts_and_skills.code_graph.server', '--port', String(PORT)];
  if (projectPath) args.push('--path', projectPath);

  console.log(`[main] Spawning: ${python} ${args.join(' ')}`);
  serverProcess = spawn(python, args, {
    cwd:   REPO_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env:   { ...process.env },
  });

  serverProcess.stdout.on('data', d => {
    try { process.stdout.write(`[server] ${d}`); } catch {}
  });
  serverProcess.stderr.on('data', d => {
    try { process.stderr.write(`[server] ${d}`); } catch {}
  });
  serverProcess.on('close', code => {
    try { console.log(`[server] exited (${code})`); } catch {}
  });

  /* Prevent EPIPE crash when launched without a console (e.g. -WindowStyle Hidden) */
  process.stdout.on('error', () => {});
  process.stderr.on('error', () => {});
}

function killServer() {
  if (!serverProcess) return;
  try { serverProcess.kill(); } catch {}
  serverProcess = null;
}

/** Poll /api/projects until the server responds (or timeout). */
function waitForServer(retries = 30, delayMs = 500) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    function attempt() {
      const req = http.get(`${SERVER_URL}/api/projects`, res => {
        if (res.statusCode === 200) return resolve();
        retry();
      });
      req.on('error', retry);
      req.setTimeout(300, () => { req.destroy(); retry(); });
    }
    function retry() {
      if (++attempts >= retries) return reject(new Error('Server did not start in time'));
      setTimeout(attempt, delayMs);
    }
    attempt();
  });
}

// ---------------------------------------------------------------------------
// Project registration (via REST, then navigate)
// ---------------------------------------------------------------------------
let mainWindow = null;

async function registerProject(projectPath) {
  addRecentProject(projectPath);
  // POST /api/register so the server starts watching this directory
  await new Promise((resolve) => {
    const body   = JSON.stringify({ path: projectPath });
    const opts   = {
      hostname: 'localhost', port: PORT,
      path: '/api/register', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    };
    const req = http.request(opts, res => { res.resume(); resolve(); });
    req.on('error', resolve);   // ignore errors — server may already know about it
    req.write(body);
    req.end();
  });

  // Navigate the renderer to the graph, passing the project as a query param
  const url = `${SERVER_URL}/?project=${encodeURIComponent(projectPath)}`;
  console.log(`[main] Navigating to ${url}`);
  mainWindow.loadURL(url);
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------
function createWindow() {
  const iconPath = path.join(__dirname, '..', 'claude_code_cli_icon-subject.ico');

  mainWindow = new BrowserWindow({
    width:           1440,
    height:          900,
    minWidth:        900,
    minHeight:       600,
    backgroundColor: '#111111',
    title:           'Code Graph',
    icon:            fs.existsSync(iconPath) ? iconPath : undefined,
    // Remove the OS titlebar — we draw our own inside index.html
    frame:           false,
    titleBarStyle:   'hidden',  // macOS: keeps traffic-lights slot hidden
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
    },
  });

  // Relay maximize / unmaximize state so the ▢/▣ button icon updates
  mainWindow.on('maximize',   () => mainWindow.webContents.send('win:state', { maximized: true  }));
  mainWindow.on('unmaximize', () => mainWindow.webContents.send('win:state', { maximized: false }));

  // Show loading screen immediately so the window doesn't flash blank
  mainWindow.loadFile(path.join(__dirname, 'loading.html'));
}

// ---------------------------------------------------------------------------
// Native menu
// ---------------------------------------------------------------------------
async function openProjectDialog() {
  const result = await dialog.showOpenDialog(mainWindow, {
    title:      'Open Project Folder',
    properties: ['openDirectory'],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    await registerProject(result.filePaths[0]);
  }
}

function buildMenu() {
  const recentProjects = loadStore().recentProjects;
  const recentItems = recentProjects.length
    ? recentProjects.map(p => ({
        label: p,
        click: () => registerProject(p),
      }))
    : [{ label: 'No recent projects', enabled: false }];

  const template = [
    {
      label: 'File',
      submenu: [
        { label: 'Open Project Folder…', accelerator: 'CmdOrCtrl+O', click: openProjectDialog },
        { label: 'Recent Projects', submenu: recentItems },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Open in Browser',
          click: () => shell.openExternal(SERVER_URL),
        },
        {
          label: 'Server API Docs',
          click: () => shell.openExternal(`${SERVER_URL}/docs`),
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------
ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title:      'Open Project Folder',
    properties: ['openDirectory'],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    await registerProject(result.filePaths[0]);
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('store:getRecentProjects', () => loadStore().recentProjects);
ipcMain.handle('store:getLastProject',    () => loadStore().lastProject);

// Window control buttons (frameless titlebar)
ipcMain.on('win:minimize', () => mainWindow?.minimize());
ipcMain.on('win:maximize', () => mainWindow?.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize());
ipcMain.on('win:close',    () => mainWindow?.close());

// ---------------------------------------------------------------------------
// JS-driven window drag (replaces -webkit-app-region:drag which is
// unreliable on Windows when served from localhost)
// ---------------------------------------------------------------------------
let _drag = null;

ipcMain.on('win:drag-start', () => {
  if (_drag?.interval) clearInterval(_drag.interval);

  // If maximized, restore first and reposition so cursor stays on the titlebar
  if (mainWindow.isMaximized()) {
    const cursor = screen.getCursorScreenPoint();
    const oldBounds = mainWindow.getBounds();
    // Remember proportional X position within the maximized window
    const ratio = (cursor.x - oldBounds.x) / oldBounds.width;
    mainWindow.unmaximize();
    const newBounds = mainWindow.getBounds();
    // Place window so the cursor stays at the same proportional X
    const newX = Math.round(cursor.x - newBounds.width * ratio);
    const newY = cursor.y - 20; // keep cursor near top of titlebar
    mainWindow.setPosition(newX, Math.max(0, newY));
  }

  _drag = {
    p0: screen.getCursorScreenPoint(),
    b0: mainWindow.getBounds(),
    interval: setInterval(() => {
      if (!_drag) return;
      const p = screen.getCursorScreenPoint();
      mainWindow.setPosition(
        Math.round(_drag.b0.x + p.x - _drag.p0.x),
        Math.round(_drag.b0.y + p.y - _drag.p0.y)
      );
    }, 16),
  };
});

ipcMain.on('win:drag-end', () => {
  if (_drag?.interval) clearInterval(_drag.interval);
  _drag = null;
});

// ---------------------------------------------------------------------------
// Custom resize (frameless windows on Windows lose OS resize handles)
// We poll cursor position on a 16ms interval while a drag is active.
// ---------------------------------------------------------------------------
let _resize = null;

ipcMain.on('win:resize-start', (_evt, dir) => {
  if (_resize?.interval) clearInterval(_resize.interval);
  _resize = {
    dir,
    b0: mainWindow.getBounds(),
    p0: screen.getCursorScreenPoint(),
    interval: setInterval(() => {
      if (!_resize) return;
      const p  = screen.getCursorScreenPoint();
      const dx = p.x - _resize.p0.x;
      const dy = p.y - _resize.p0.y;
      let { x, y, width, height } = _resize.b0;
      const d = _resize.dir;
      if (d.includes('e')) width  = Math.max(900, _resize.b0.width  + dx);
      if (d.includes('s')) height = Math.max(600, _resize.b0.height + dy);
      if (d.includes('w')) { x = _resize.b0.x + dx; width  = Math.max(900, _resize.b0.width  - dx); }
      if (d.includes('n')) { y = _resize.b0.y + dy; height = Math.max(600, _resize.b0.height - dy); }
      mainWindow.setBounds({ x: Math.round(x), y: Math.round(y), width: Math.round(width), height: Math.round(height) });
    }, 16),
  };
});

ipcMain.on('win:resize-end', () => {
  if (_resize?.interval) clearInterval(_resize.interval);
  _resize = null;
});

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------
app.whenReady().then(async () => {
  buildMenu();
  createWindow();

  // Initial project: CLI arg > last used project
  const cliPath     = process.argv.find((a, i) => i > 1 && !a.startsWith('-'));
  const store       = loadStore();
  const initialPath = cliPath || store.lastProject || null;

  startServer(initialPath);

  try {
    await waitForServer();
  } catch (err) {
    console.error('[main] Server failed to start:', err.message);
    mainWindow.loadURL(
      `data:text/html,<body style="background:#111;color:#f55;font-family:monospace;padding:2em">` +
      `<h2>Server failed to start</h2><pre>${err.message}</pre>` +
      `<p>Make sure Python deps are installed:<br>` +
      `<code>pip install fastapi uvicorn watchdog</code></p></body>`
    );
    return;
  }

  if (initialPath) {
    await registerProject(initialPath);
  } else {
    // No project configured — load the server UI and prompt for a folder
    mainWindow.loadURL(SERVER_URL);
    // Give the page a moment to render, then ask for a project
    setTimeout(openProjectDialog, 800);
  }
});

app.on('before-quit', killServer);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
