"""Quick static server for the contact map. Run from sonar_web/:
    python serve.py [port]
Then open http://localhost:<port>/index.html
Point ContactExporter(out_dir=...) at the same "data" folder this serves.
"""
import http.server, socketserver, sys, os

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Serving sonar_web at http://localhost:{port}/index.html")
    httpd.serve_forever()
