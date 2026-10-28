local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrApplicationView = import 'LrApplicationView'
local BASE = '/Users/rohan/Projects/trichroma/research/upright/lrtest'
local DIRS = { BASE .. '/nm', BASE .. '/in' }
local function trace(m)
    local f = io.open(BASE .. '/TRACE_NM.txt', 'a')
    if f then f:write(os.date('%H:%M:%S') .. '  ' .. tostring(m) .. '\n'); f:close() end
end
local function apply(catalog, p, tbl, tag)
    local ok = LrTasks.pcall(function()
        catalog:withWriteAccessDo('u', function() p:applyDevelopSettings(tbl) end, { timeout = 20 })
    end)
    trace('  ' .. tag .. ' ok=' .. tostring(ok))
end
LrTasks.startAsyncTask(function()
    trace('=== non-manhattan probe ===')
    local catalog = LrApplication.activeCatalog()
    local names = {}
    for _, DIR in ipairs(DIRS) do
        local pipe = io.popen('ls "' .. DIR .. '" | grep tif')
        if pipe then
            for l in pipe:lines() do
                local n = l:gsub('%.tif$','')
                if #n > 0 then names[#names+1] = { DIR .. '/' .. n .. '.tif', n } end
            end
            pipe:close()
        end
    end
    trace('found ' .. #names)
    local photos = {}
    LrTasks.pcall(function()
        catalog:withWriteAccessDo('i', function()
            for _, e in ipairs(names) do
                local p, n = e[1], e[2]
                local got = catalog:findPhotoByPath(p)
                if not got then local h = io.open(p,'r'); if h then h:close(); got = catalog:addPhoto(p) end end
                if got then photos[#photos+1] = {got, n} end
            end
        end)
    end)
    trace('imported ' .. #photos)
    local col
    LrTasks.pcall(function()
        catalog:withWriteAccessDo('c', function()
            col = catalog:createCollection('nm probe', nil, true)
            local l = {}; for _, x in ipairs(photos) do l[#l+1] = x[1] end
            if col then col:addPhotos(l) end
        end)
    end)
    if col then LrTasks.pcall(function() catalog:setActiveSources({col}) end) end
    LrApplicationView.switchToModule('develop')
    LrTasks.sleep(4)
    for _, x in ipairs(photos) do
        local p, nm = x[1], x[2]
        LrTasks.pcall(function() catalog:setSelectedPhotos(p, {p}) end)
        LrTasks.sleep(1.5)
        apply(catalog, p, { PerspectiveUpright = 0 }, nm .. ' reset')
        LrTasks.sleep(1.0)
        apply(catalog, p, { PerspectiveUpright = 1 }, nm .. ' auto')
        LrTasks.sleep(10)
        local s = p:getDevelopSettings()
        trace(string.format('%s AUTO f35=%s', nm, tostring(s.UprightFocalLength35mm)))
        apply(catalog, p, { PerspectiveUpright = 2 }, nm .. ' full')
        LrTasks.sleep(12)
        local s2 = p:getDevelopSettings()
        trace(string.format('%s FULL f35=%s', nm, tostring(s2.UprightFocalLength35mm)))
    end
    trace('=== nm done ===')
end)
