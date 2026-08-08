import os
import sys
import socket
import webbrowser
import http.server
import socketserver
import threading
import time

DIRECTORY = os.path.abspath(os.path.dirname(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def find_free_port(preferred_ports=[8085, 8088, 8090, 8888, 9000]):
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def start_gui():
    print("\n==========================================================================")
    print(" LAUNCHING SATELLITE CROP HEALTH & DROUGHT MONITORING GUI DASHBOARD ")
    print("==========================================================================")
    
    port = find_free_port()
    url = f"http://localhost:{port}/dashboard/index.html"
    
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        
    def serve():
        try:
            with ReusableTCPServer(("", port), Handler) as httpd:
                httpd.serve_forever()
        except Exception as e:
            pass
            
    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    
    time.sleep(0.5)
    print(f"\nSUCCESS: GUI Server Running at: {url}")
    print(f"Opening browser at: {url}\n")
    
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Browser launch note: {e}")

    dashboard_path = os.path.abspath(os.path.join(DIRECTORY, "dashboard", "index.html"))
    print(f"Direct Local File Path: file:///{dashboard_path.replace('\\', '/')}\n")
    print("GUI is running! Press Ctrl+C in terminal to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nGUI Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    start_gui()
