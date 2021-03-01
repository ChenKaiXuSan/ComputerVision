import platform

def getSystemName():
    '''
    判断当前运行环境使用的是什么操作系统

    Returns:
        str: 返回当前使用操作系统的str
    '''    
    sys_name = platform.system()
    if(sys_name == 'Windows'):
        print("now is window")
        return sys_name
    elif(sys_name == 'Linux'):
        print("now is Linux")
        return sys_name
    else:
        print('other system name,' + sys_name)
