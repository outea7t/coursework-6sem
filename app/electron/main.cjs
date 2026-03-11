const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

let mainWindow = null;
let bridge = null;

const PROJECT_ROOT = path.join(__dirname, "..", "..");
const BRIDGE_PATH = path.join(PROJECT_ROOT, "bridge.py");

// Detect dev mode: Vite dev server running
const isDev = !fs.existsSync(path.join(__dirname, "..", "dist", "index.html"));

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 520,
    height: 680,
    resizable: false,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: "#f8f8fa",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function startBridge() {
  bridge = spawn("python3", [BRIDGE_PATH], {
    cwd: PROJECT_ROOT,
    stdio: ["pipe", "pipe", "pipe"],
  });

  let buffer = "";

  bridge.stdout.on("data", (data) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete last line

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);

        // Convert image file to base64 data URL
        if (event.image && fs.existsSync(event.image)) {
          const imgBuffer = fs.readFileSync(event.image);
          const ext = event.image.endsWith(".jpg") ? "jpeg" : "png";
          event.imageData =
            `data:image/${ext};base64,` + imgBuffer.toString("base64");
        }

        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("bridge-event", event);
        }
      } catch {
        // ignore parse errors
      }
    }
  });

  bridge.stderr.on("data", (data) => {
    // Log Python stderr (tqdm, logging) to console for debugging
    const text = data.toString();
    if (text.trim()) {
      process.stderr.write("[bridge] " + text);
    }
  });

  bridge.on("close", (code) => {
    console.log(`Bridge exited with code ${code}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("bridge-event", {
        type: "error",
        message: `Python process exited (code ${code}). Restart the app.`,
      });
    }
    bridge = null;
  });
}

function sendToBridge(data) {
  if (bridge && bridge.stdin.writable) {
    bridge.stdin.write(JSON.stringify(data) + "\n");
  }
}

// IPC handlers
ipcMain.handle("generate", (_event, params) => {
  sendToBridge({ type: "generate", ...params });
});

ipcMain.handle("ping", () => {
  sendToBridge({ type: "ping" });
});

app.whenReady().then(() => {
  createWindow();
  startBridge();
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("before-quit", () => {
  if (bridge) {
    bridge.kill("SIGTERM");
    bridge = null;
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
