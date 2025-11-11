"""
Real-time Network Traffic Capture Module
Captures live network packets and extracts features for IDS analysis
"""
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, ICMP
from collections import defaultdict
from datetime import datetime
import threading
import queue
from typing import Dict, List, Optional
import time


class NetworkCapture:
    """
    Captures live network traffic and extracts UNSW-NB15 compatible features
    """

    def __init__(self, interface: Optional[str] = None, packet_count: int = 100):
        """
        Initialize network capture

        Args:
            interface: Network interface to capture from (None = default)
            packet_count: Number of packets to capture before processing
        """
        self.interface = interface
        self.packet_count = packet_count
        self.packets = []
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'end_time': None,
            'bytes_sent': 0,
            'bytes_received': 0
        })
        self.packet_queue = queue.Queue()
        self.is_capturing = False
        self.capture_thread = None

    def get_flow_id(self, packet) -> str:
        """Generate unique flow ID from packet"""
        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            sport = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0)
            dport = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
            proto = packet[IP].proto
            # Bidirectional flow ID
            if src < dst:
                return f"{src}:{sport}-{dst}:{dport}:{proto}"
            else:
                return f"{dst}:{dport}-{src}:{sport}:{proto}"
        return "unknown"

    def packet_callback(self, packet):
        """Callback function for each captured packet"""
        if IP in packet:
            self.packets.append(packet)
            flow_id = self.get_flow_id(packet)
            flow = self.flows[flow_id]

            # Update flow information
            if flow['start_time'] is None:
                flow['start_time'] = datetime.now()
            flow['end_time'] = datetime.now()
            flow['packets'].append(packet)

            # Track bytes
            if hasattr(packet, 'len'):
                flow['bytes_sent'] += packet.len

            # Add to queue for processing
            self.packet_queue.put(packet)

    def extract_features(self, packet) -> Dict:
        """
        Extract UNSW-NB15 compatible features from packet
        Returns a dictionary of features
        """
        features = {}

        # IP layer features
        if IP in packet:
            ip = packet[IP]
            features['srcip'] = ip.src
            features['sport'] = 0
            features['dstip'] = ip.dst
            features['dsport'] = 0
            features['proto'] = ip.proto
            features['dur'] = 0.0  # Will be calculated from flow
            features['sbytes'] = len(packet)
            features['dbytes'] = 0  # Will be calculated from flow
            features['sttl'] = ip.ttl
            features['dttl'] = 0
            features['sloss'] = 0
            features['dloss'] = 0
            features['sload'] = 0.0
            features['dload'] = 0.0
            features['spkts'] = 1
            features['dpkts'] = 0

            # TCP features
            if TCP in packet:
                tcp = packet[TCP]
                features['sport'] = tcp.sport
                features['dsport'] = tcp.dport
                features['state'] = self._get_tcp_state(tcp)
                features['swin'] = tcp.window
                features['dwin'] = 0
                features['stcpb'] = tcp.seq
                features['dtcpb'] = tcp.ack
                features['tcprtt'] = 0.0
                features['synack'] = 0.0
                features['ackdat'] = 0.0
            elif UDP in packet:
                udp = packet[UDP]
                features['sport'] = udp.sport
                features['dsport'] = udp.dport
                features['state'] = 'CON'
            else:
                features['state'] = 'INT'

            # Additional features
            features['smean'] = len(packet)
            features['dmean'] = 0
            features['trans_depth'] = 0
            features['res_bdy_len'] = 0
            features['sjit'] = 0.0
            features['djit'] = 0.0
            features['stime'] = time.time()
            features['ltime'] = time.time()
            features['sintpkt'] = 0.0
            features['dintpkt'] = 0.0
            features['tcprtt'] = 0.0
            features['synack'] = 0.0
            features['ackdat'] = 0.0
            features['is_sm_ips_ports'] = 0
            features['ct_state_ttl'] = 0
            features['ct_flw_http_mthd'] = 0
            features['is_ftp_login'] = 0
            features['ct_ftp_cmd'] = 0
            features['ct_srv_src'] = 0
            features['ct_srv_dst'] = 0
            features['ct_dst_ltm'] = 0
            features['ct_src_ltm'] = 0
            features['ct_src_dport_ltm'] = 0
            features['ct_dst_sport_ltm'] = 0
            features['ct_dst_src_ltm'] = 0

            # Service detection (simplified)
            features['service'] = self._detect_service(features.get('dsport', 0))

        return features

    def _get_tcp_state(self, tcp) -> str:
        """Determine TCP connection state"""
        flags = tcp.flags
        if flags & 0x02:  # SYN
            return 'CON' if flags & 0x10 else 'REQ'
        elif flags & 0x01:  # FIN
            return 'FIN'
        elif flags & 0x04:  # RST
            return 'RST'
        elif flags & 0x10:  # ACK
            return 'CON'
        return 'INT'

    def _detect_service(self, port: int) -> str:
        """Detect service based on port number"""
        services = {
            20: 'ftp-data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'dns', 80: 'http', 110: 'pop3',
            143: 'imap', 443: 'https', 3389: 'rdp', 3306: 'mysql',
            5432: 'postgresql', 6379: 'redis', 27017: 'mongodb'
        }
        return services.get(port, '-')

    def start_capture(self, duration: Optional[int] = None, count: Optional[int] = None):
        """
        Start capturing packets

        Args:
            duration: Capture duration in seconds (None = infinite)
            count: Number of packets to capture (None = infinite)
        """
        self.is_capturing = True
        print(f"Starting packet capture on interface: {self.interface or 'default'}")
        print(f"Press Ctrl+C to stop...")

        try:
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                timeout=duration,
                count=count,
                store=False
            )
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        finally:
            self.is_capturing = False
            print(f"Captured {len(self.packets)} packets")

    def start_capture_async(self, duration: Optional[int] = None, count: Optional[int] = None):
        """Start capture in background thread"""
        if self.is_capturing:
            print("Capture already running")
            return

        self.capture_thread = threading.Thread(
            target=self.start_capture,
            args=(duration, count),
            daemon=True
        )
        self.capture_thread.start()

    def stop_capture(self):
        """Stop packet capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)

    def get_recent_packets(self, n: int = 10) -> List[Dict]:
        """Get N most recent packets with extracted features"""
        recent = self.packets[-n:] if len(self.packets) >= n else self.packets
        return [self.extract_features(pkt) for pkt in recent]

    def get_flow_statistics(self) -> pd.DataFrame:
        """Get flow-level statistics"""
        flow_stats = []
        for flow_id, flow_data in self.flows.items():
            if flow_data['start_time'] and flow_data['end_time']:
                duration = (flow_data['end_time'] - flow_data['start_time']).total_seconds()
                flow_stats.append({
                    'flow_id': flow_id,
                    'packets': len(flow_data['packets']),
                    'duration': duration,
                    'bytes': flow_data['bytes_sent'],
                    'rate': flow_data['bytes_sent'] / duration if duration > 0 else 0
                })

        return pd.DataFrame(flow_stats)

    def clear(self):
        """Clear captured packets and flows"""
        self.packets.clear()
        self.flows.clear()
        while not self.packet_queue.empty():
            self.packet_queue.get()


if __name__ == "__main__":
    # Test the network capture
    capture = NetworkCapture()

    print("Testing network capture (capturing 50 packets)...")
    capture.start_capture(count=50)

    print(f"\nCaptured {len(capture.packets)} packets")

    # Show recent packets
    recent = capture.get_recent_packets(5)
    print("\nRecent packets:")
    for i, pkt_features in enumerate(recent, 1):
        print(f"{i}. {pkt_features.get('srcip', 'N/A')}:{pkt_features.get('sport', 'N/A')} -> "
              f"{pkt_features.get('dstip', 'N/A')}:{pkt_features.get('dsport', 'N/A')} "
              f"[{pkt_features.get('service', 'unknown')}]")

    # Show flow statistics
    flows = capture.get_flow_statistics()
    if not flows.empty:
        print("\nFlow statistics:")
        print(flows.head())
