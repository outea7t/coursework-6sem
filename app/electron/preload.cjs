// preload-скрипт: пробрасывает безопасный api моста в window.bridge,
// чтобы React мог звать generate/ping и подписываться на события

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("bridge", {
  generate: (params) => ipcRenderer.invoke("generate", params),
  ping: () => ipcRenderer.invoke("ping"),
  onEvent: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("bridge-event", handler);
    return () => ipcRenderer.removeListener("bridge-event", handler);
  },
});
