"""
PCAP File Analyzer Module
Analyzes PCAP files and extracts features for IDS detection
"""
import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from pathlib import Path
from typing import List, Dict, Optional
from src.network_capture import NetworkCapture


class PCAPAnalyzer:
    """Analyzes PCAP files for intrusion detection"""

    def __init__(self, pcap_file: str):
        """
        Initialize PCAP analyzer

        Args:
            pcap_file: Path to PCAP file
        """
        self.pcap_file = Path(pcap_file)
        self.packets = []
        self.features_df = None
        self.capture = NetworkCapture()

    def load_pcap(self) -> int:
        """
        Load PCAP file

        Returns:
            Number of packets loaded
        """
        print(f"Loading PCAP file: {self.pcap_file}")
        try:
            self.packets = rdpcap(str(self.pcap_file))
            print(f"Loaded {len(self.packets)} packets")
            return len(self.packets)
        except Exception as e:
            print(f"Error loading PCAP: {e}")
            return 0

    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract features from all packets in PCAP

        Returns:
            DataFrame with extracted features
        """
        if not self.packets:
            self.load_pcap()

        features_list = []
        print(f"Extracting features from {len(self.packets)} packets...")

        for i, packet in enumerate(self.packets):
            if IP in packet:
                features = self.capture.extract_features(packet)
                features['packet_id'] = i
                features_list.append(features)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.packets)} packets")

        self.features_df = pd.DataFrame(features_list)
        print(f"Extracted features from {len(self.features_df)} packets")
        return self.features_df

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of PCAP file"""
        if self.features_df is None:
            self.extract_all_features()

        stats = {
            'total_packets': len(self.features_df),
            'unique_src_ips': self.features_df['srcip'].nunique(),
            'unique_dst_ips': self.features_df['dstip'].nunique(),
            'protocols': self.features_df['proto'].value_counts().to_dict(),
            'top_services': self.features_df['service'].value_counts().head(10).to_dict(),
            'total_bytes': self.features_df['sbytes'].sum(),
            'duration': (self.features_df['ltime'].max() - self.features_df['stime'].min())
        }

        return stats

    def get_protocol_distribution(self) -> pd.DataFrame:
        """Get protocol distribution"""
        if self.features_df is None:
            self.extract_all_features()

        proto_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        proto_dist = self.features_df['proto'].value_counts()
        proto_dist.index = proto_dist.index.map(lambda x: proto_map.get(x, f'Other ({x})'))

        return proto_dist.to_frame('count')

    def get_top_talkers(self, n: int = 10) -> pd.DataFrame:
        """Get top N source IPs by packet count"""
        if self.features_df is None:
            self.extract_all_features()

        return self.features_df['srcip'].value_counts().head(n).to_frame('packets')

    def get_service_distribution(self, n: int = 10) -> pd.DataFrame:
        """Get top N services"""
        if self.features_df is None:
            self.extract_all_features()

        return self.features_df['service'].value_counts().head(n).to_frame('count')


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pcap_analyzer.py <pcap_file>")
        sys.exit(1)

    analyzer = PCAPAnalyzer(sys.argv[1])
    analyzer.load_pcap()

    print("\n=== Summary Statistics ===")
    stats = analyzer.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Protocol Distribution ===")
    print(analyzer.get_protocol_distribution())

    print("\n=== Top Talkers ===")
    print(analyzer.get_top_talkers())

    print("\n=== Service Distribution ===")
    print(analyzer.get_service_distribution())
