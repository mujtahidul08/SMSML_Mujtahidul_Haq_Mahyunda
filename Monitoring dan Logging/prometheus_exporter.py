import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Menginisiasi Metriks (Target minimal 10 metriks untuk Advance)
CPU_USAGE = Gauge('system_cpu_usage', 'Persentase penggunaan CPU')
MEMORY_USAGE = Gauge('system_memory_usage', 'Persentase penggunaan Memory')
REQUEST_COUNT = Counter('model_request_total', 'Total request ke model API')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Durasi inferensi model')
MODEL_ACCURACY = Gauge('model_live_accuracy', 'Akurasi model di fase produksi')
ERROR_COUNT = Counter('model_error_total', 'Total error pada request')
ACTIVE_USERS = Gauge('model_active_users', 'Jumlah pengguna aktif saat ini')
DISK_IO_READ = Counter('system_disk_io_read_bytes', 'Total read bytes disk')
DISK_IO_WRITE = Counter('system_disk_io_write_bytes', 'Total write bytes disk')
NETWORK_TRAFFIC = Gauge('system_network_traffic_mbps', 'Trafik jaringan dalam Mbps')

if __name__ == '__main__':
    # Berjalan di port 8000 sesuai target prometheus.yml
    start_http_server(8000)
    print("[INFO] Prometheus Exporter berjalan di port 8000...")
    
    while True:
        # Simulasi fluktuasi metrik produksi
        CPU_USAGE.set(random.uniform(20.0, 85.0))
        MEMORY_USAGE.set(random.uniform(40.0, 90.0))
        REQUEST_COUNT.inc(random.randint(1, 5))
        REQUEST_LATENCY.observe(random.uniform(0.05, 0.5))
        MODEL_ACCURACY.set(random.uniform(0.82, 0.95))
        ACTIVE_USERS.set(random.randint(10, 150))
        DISK_IO_READ.inc(random.randint(1024, 8192))
        DISK_IO_WRITE.inc(random.randint(512, 4096))
        NETWORK_TRAFFIC.set(random.uniform(1.5, 25.0))
        
        if random.random() < 0.05:
            ERROR_COUNT.inc()
            
        time.sleep(2)