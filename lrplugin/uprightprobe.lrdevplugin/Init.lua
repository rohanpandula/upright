local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrApplicationView = import 'LrApplicationView'
local BASE = '/Users/rohan/Projects/trichroma/research/upright/lrtest'
local SWEEP = BASE .. '/sweep'
local NAMES = {'P05','P10','P15','P20','P25','P30','P35',
               'Y03','Y06','Y09','Y12','Y15','Y18','Y24'}
local function trace(m)
    local f = io.open(BASE .. '/TRACE.txt', 'a')
    if f then f:write(os.date('%H:%M:%S') .. '  ' .. tostring(m) .. '\n'); f:close() end
end
LrTasks.startAsyncTask(function()
    trace('=== sweep start ===')
    local catalog = LrApplication.activeCatalog()
    local photos = {}
    for _, n in ipairs(NAMES) do
        local p = SWEEP .. '/' .. n .. '.tif'
        local f = io.open(p, 'r')
        if f then
            f:close()
            local got = catalog:findPhotoByPath(p)
            if got then photos[#photos+1] = got end
        else
            trace('MISSING ' .. p)
        end
    end
    trace('pre-existing ' .. #photos)

    local col
    local ok, err = LrTasks.pcall(function()
        catalog:withWriteAccessDo('upright sweep import', function()
            for _, n in ipairs(NAMES) do
                local p = SWEEP .. '/' .. n .. '.tif'
                if not catalog:findPhotoByPath(p) then
                    local np = catalog:addPhoto(p)
                    if np then photos[#photos+1] = np end
                end
            end
            col = catalog:createCollection('upright sweep', nil, true)
            if col then col:addPhotos(photos) end
        end)
    end)
    trace('write ok=' .. tostring(ok) .. ' err=' .. tostring(err) .. ' photos=' .. #photos)

    if col then
        local ok2 = LrTasks.pcall(function() catalog:setActiveSources({col}) end)
        trace('setActiveSources ok=' .. tostring(ok2))
    end
    LrApplicationView.switchToModule('develop')
    LrTasks.sleep(3)

    for _, p in ipairs(photos) do
        local nm = p:getFormattedMetadata('fileName')
        LrTasks.pcall(function() catalog:setSelectedPhotos(p, {p}) end)
        LrTasks.sleep(1.5)
        local t = catalog:getTargetPhoto()
        local tn = t and t:getFormattedMetadata('fileName') or 'NONE'
        LrTasks.pcall(function() p:applyDevelopSettings{ PerspectiveUpright = 0 } end)
        LrTasks.sleep(1.0)
        LrTasks.pcall(function() p:applyDevelopSettings{ PerspectiveUpright = 1 } end)
        LrTasks.sleep(9)
        local s = p:getDevelopSettings()
        trace(string.format('%s target=%s f35=%s', nm, tn, tostring(s.UprightFocalLength35mm)))
    end
    trace('=== sweep done ===')
end)
