import csv
import threading
import time
from scapy.all import sniff, IP, TCP, UDP


class Sniffer:
    def __init__(self):
        self.flows = {}
        self.lock = threading.Lock()
        self.is_running = False

    def handler(self, packet):
        if IP not in packet:
            return
        src = packet[IP].src
        dst = packet[IP].dst
        numb = {6: 'TCP', 17: 'UDP'}
        proto = numb.get(packet[IP].proto, 'OTHER')
        src_port = 0
        dst_port = 0
        tcp_flags = ''
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = str(packet[TCP].flags)
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        key = (src, dst, proto, src_port, dst_port)
        packet_len = len(packet)
        now_time = time.time()

        with self.lock:
            if key not in self.flows:
                self.flows[key] = {
                    'start': now_time,
                    'last': now_time,
                    'pkts': 1,
                    'proto': proto,
                    'bytes': packet_len,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'tcp_flags': set(tcp_flags),
                }
            else:
                self.flows[key]['last'] = now_time
                self.flows[key]['pkts'] += 1
                self.flows[key]['bytes'] += packet_len
                if tcp_flags:
                    self.flows[key]['tcp_flags'] |= set(tcp_flags)

    def export(self, force=False):
        exporter = []
        keys_to_delete = []
        now_time = time.time()
        with self.lock:
            for key, data in self.flows.items():
                duration = data['last'] - data['start']
                time_last_packet = now_time - data['last']
                if force or time_last_packet > 3 or duration > 15:
                    flow_record = [
                        data['start'],
                        key[0], key[1], key[2], key[3], key[4],
                        round(duration, 3),
                        data['pkts'],
                        data['bytes'],
                        ''.join(sorted(data['tcp_flags'])),
                    ]
                    exporter.append(flow_record)
                    keys_to_delete.append(key)
            for k in keys_to_delete:
                del self.flows[k]
        return exporter

    def run(self):
        self.is_running = True
        def sniff_loop():
            sniff(prn=self.handler, store=False,
                  stop_filter=lambda p: not self.is_running)

        threading.Thread(target=sniff_loop, daemon=True).start()

    def stop(self):
        self.is_running = False

#debugging code
HEADER = ['timestamp', 'src', 'dst', 'proto', 'src_port', 'dst_port',
          'duration', 'packets', 'bytes', 'tcp_flags']

def collect(file, duration_sec):
    snif = Sniffer()
    with open(file, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(HEADER)
    snif.run()
    start_time = time.time()
    tick = 0
    while time.time() - start_time < duration_sec:
        time.sleep(60)
        tick += 1
        flows = snif.export()
        if flows:
            with open(file, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(flows)
        if tick % 5 == 0:
            rem_h = round((duration_sec - (time.time() - start_time)) / 3600, 2)
            print(f'осталось: {rem_h} ч, флоу в этой пачке: {len(flows)}')
    snif.stop()

    tail_flows = snif.export(force=True)
    if tail_flows:
        with open(file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(tail_flows)

    print('сбор окончен')



