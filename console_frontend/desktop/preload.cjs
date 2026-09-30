const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('nerfDesktop', {
  minimize: () => ipcRenderer.send('desktop:minimize'),
  maximize: () => ipcRenderer.send('desktop:maximize'),
  close: () => ipcRenderer.send('desktop:close'),
  setFullscreen: value => ipcRenderer.invoke('desktop:setFullscreen', value),
  isMaximized: () => ipcRenderer.invoke('desktop:isMaximized'),
  onMaximized: callback => {
    const listener = (_event, value) => callback(Boolean(value))
    ipcRenderer.on('desktop:maximized', listener)
    return () => ipcRenderer.removeListener('desktop:maximized', listener)
  },
})
