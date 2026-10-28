with timeout of 8 seconds
  tell application "System Events"
    tell process "Adobe Lightroom Classic"
      repeat with w in windows
        set hit to my findRole(w, "Guided", "AXButton")
        if hit is not missing value then
          perform action "AXPress" of hit
          return "pressed"
        end if
      end repeat
      return "missing"
    end tell
  end tell
end timeout

on findRole(root, needle, wantRole)
  try
    set kids to UI elements of root
  on error
    return missing value
  end try
  repeat with k in kids
    try
      if (role of k as string) is wantRole then
        try
          if (name of k as string) is needle then return contents of k
        end try
        try
          if (description of k as string) is needle then return contents of k
        end try
      end if
    end try
    try
      set r to my findRole(k, needle, wantRole)
      if r is not missing value then return r
    end try
  end repeat
  return missing value
end findRole
