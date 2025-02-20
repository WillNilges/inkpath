local function getLibPath(libName)
    local isWindows = package.config:sub(1, 1) == "\\"
    local scriptDir = debug.getinfo(1, "S").source:sub(2)
    local pluginDir = scriptDir:match(isWindows and "(.*\\)" or "(.*/)")
    local libPath = pluginDir .. libName .. "." .. (isWindows and "dll" or "so")
    return libPath
end

return getLibPath
