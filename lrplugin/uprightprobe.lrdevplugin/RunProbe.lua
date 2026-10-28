local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrApplicationView = import 'LrApplicationView'
local BASE = '/Users/rohan/Projects/trichroma/research/upright/lrtest'
local TESTIMG = BASE .. '/in/A_clean.tif'
local function trace(m)
    local f = io.open(BASE .. '/TRACE.txt', 'a')
    if f then f:write(os.date('%H:%M:%S') .. '  ' .. tostring(m) .. '\n'); f:close() end
end
local function name(p) return p and p:getFormattedMetadata('fileName') or 'NONE' end
local function expo(p)
    local ok, s = LrTasks.pcall(function() return p:getDevelopSettings() end)
    return (ok and s) and tostring(s.Exposure) or '?'
end
LrTasks.startAsyncTask(function()
    trace('=== mechanism ===')
    local catalog = LrApplication.activeCatalog()
    local t = catalog:getTargetPhoto()
    trace('active=' .. name(t) .. ' exposure=' .. (t and expo(t) or '-'))
    if t then
        LrTasks.pcall(function() t:applyDevelopSettings{ Exposure = 1.0 } end)
        LrTasks.sleep(3)
        trace('after apply to ACTIVE: exposure=' .. expo(t))
    end
    LrApplicationView.switchToModule('library')
    LrTasks.sleep(2)
    local p = catalog:findPhotoByPath(TESTIMG)
    trace('testphoto=' .. name(p))
    LrTasks.pcall(function() catalog:setSelectedPhotos(p, {}) end)
    LrTasks.sleep(3)
    trace('after setSel: target=' .. name(catalog:getTargetPhoto()))
    trace('=== done ===')
end)
