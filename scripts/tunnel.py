"""
Sentiment Analysis – Public Live Tunnel
Exposes local Flask server (http://127.0.0.1:5000) to a public HTTPS URL.
"""
import subprocess, re, time, sys, os

TUNNEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PUBLIC_URL.txt")

def start_tunnel():
    cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30', '-R', '80:127.0.0.1:5000', 'nokey@localhost.run']
    print("Starting SSH tunnel via localhost.run...")
    sys.stdout.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    url = None
    for line in iter(proc.stdout.readline, ''):
        line_clean = line.strip()
        if line_clean:
            print(line_clean)
            sys.stdout.flush()
        match = re.search(r'https://[a-zA-Z0-9\.\-]+lhr\.life', line)
        if match:
            url = match.group(0)
            print("\n" + "="*60)
            print(f"🚀 LIVE PUBLIC ACCESS URL: {url}")
            print("="*60 + "\n")
            sys.stdout.flush()
            with open(TUNNEL_FILE, "w", encoding="utf-8") as f:
                f.write(url)
    
    proc.wait()

if __name__ == "__main__":
    while True:
        try:
            start_tunnel()
        except Exception as e:
            print(f"Tunnel error: {e}, restarting in 5s...")
            sys.stdout.flush()
            time.sleep(5)
