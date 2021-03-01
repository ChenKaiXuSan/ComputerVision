import platform

def getSystemName():
    sys_name = platform.system()
    if(sys_name == 'Windows'):
        print("now is window")
        return sys_name
    elif(sys_name == 'Linux'):
        print("now is Linux")
        return sys_name
    else:
        print('other system name,' + sys_name)
