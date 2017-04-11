local utils = {}

function utils.getArgs(args, name, default)
  if args == nil then args = {} end
  if args[name] == nil and default == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif args[name] == nil then
    return default
  else
    return args[name]
  end
end

--[[
  Prints the time and a message in the form of

  <time> <message>

  example: 08:58:23 Hello World!
]]--
function utils.printTime(message)
  local timeObject = os.date("*t")
  local currTime = ("%02d:%02d:%02d"):format(timeObject.hour, timeObject.min, timeObject.sec)
  print(string.format("%s %s", currTime, message))
end

return utils