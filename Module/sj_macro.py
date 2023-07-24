
import pywinauto

def show_window_process():
    for e in pywinauto.findwindows.find_elements():
        name, proces_id = e.name, e.process_id
        print("Unknown" if name == "" else name, ", ", "pid: ", proces_id)

def find_process(search_name):
    for e in pywinauto.findwindows.find_elements():
        name, proces_id = e.name, e.process_id
        if search_name in name:
            return name, proces_id

    raise Exception("Cant find process name!: " + search_name)

def control_identifiers(app_control):
    app_control.print_control_identifiers()

def launch_app(back_end = "uia"):
    # Win32 API (backend="win32")
    # MS UI Automation (backend="uia")
    pwa_app = pywinauto.application.Application(backend = back_end)
    return pwa_app

def access_window_control(app, name):
    return app[name]
    
def connect(app, pid):
    app.connect(process = pid)


