import psutil
import csv
import time
import subprocess
import re


def exec_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    return result.stdout.strip()

def get_temperature():
    output = exec_command("vcgencmd measure_temp")
    temp_str = re.search(r'\d+\.\d+', output).group(0)
    return float(temp_str)

def get_used_ram():
    output = exec_command("free -m | grep Mem")
    used_ram = int(output.split()[2])  # Get the used memory in MB
    return used_ram

def get_cpu_usage():
    output = exec_command("top -bn1 | grep 'Cpu(s)'")
    usage_str = re.search(r'\d+\.\d+', output).group(0)
    cpu_usage = 100 - float(usage_str)  # Get idle CPU, subtract from 100 to get usage
    return cpu_usage


# Usage examples

class SystemMetricsCSV:
    def __init__(self, output_file='system_metrics.csv'):
        self.output_file = output_file
        # Initialize CSV file with headers
        with open(self.output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Inference time(ms)', 'Timestamp', 'Used RAM (MB)', 'CPU Usage (%)', 'Temperature (°C)'])

    def get_used_ram(self):
        ram = psutil.virtual_memory()
        return ram.used / (1024 * 1024)  # Convert to MB

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1, percpu=True)

    def get_temperature(self):
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as file:
            temp = float(file.read()) / 1000  # Convert from millidegree to degree
        return temp

    def write_metrics(self, name, fps):
        with open(self.output_file, 'a', newline='') as file:

            writer = csv.writer(file)
            temperature = get_temperature()
            ram_usage = get_used_ram()
            # cpu_usage = get_cpu_usage()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([name, fps, timestamp, ram_usage, 0, temperature])
