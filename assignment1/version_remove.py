import os, sys, platform

print(os.getcwd())

def getlibname(s):
    return s.split("==")[0]

with open("./assignment1/requirements.txt", 'r') as f:
    lines = f.readlines()
    lines = list(map(getlibname, lines) )

if platform.system().lower() == "windows":
    try:
        lines[lines.index("gnureadline")] = "pyreadline"
    except ValueError as e:
        print("Warning: ", e)
    
with open("./assignment1/requirements_no_version.txt", 'w') as f:
    for line in lines:
        print(line, file=f)

