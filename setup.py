import cx_Freeze

import sys


base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables=[cx_Freeze.Executable("main.py",icon="logo-UNLa.ico",base=base,)]



cx_Freeze.setup(
 
	version="1.0",   
	name="Reconocimiento Facial",
     
	options={"build_exe":{"include_files":["view","resources"]}},
    
	description=" Sistema de Reconocimiento Facial",
    	
	executables=executables

    
)