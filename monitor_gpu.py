import time
import os
import subprocess

def monitor():
    try:
        while True:
            os.system('clear')
            subprocess.run(['nvidia-smi'])
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    monitor()
