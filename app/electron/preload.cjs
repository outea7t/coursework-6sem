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
