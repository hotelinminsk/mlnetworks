import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import SIMULATED_SOURCE_POOL, SIMULATED_TARGET_POOL


class TrafficReplay:
    """Replays UNSW-NB15 (or uploaded) data as if it were live network traffic."""

    def __init__(self, data_path: str, speed_multiplier: float = 1.0, data_frame: Optional[pd.DataFrame] = None):
        self.data_path = Path(data_path)
        self.speed_multiplier = speed_multiplier
        self.custom_data = data_frame
        self.data = None
        self.current_index = 0
        self.is_playing = False
        self.play_thread = None
        self.packet_queue = queue.Queue()
        self.attack_log = []
        self.source_pool = SIMULATED_SOURCE_POOL.copy() or ["192.168.1.10"]
        self.target_pool = SIMULATED_TARGET_POOL.copy() or ["10.0.0.10"]
        self._source_idx = 0
        self._target_idx = 0

    def load_data(self) -> bool:
        try:
            if self.custom_data is not None:
                self.data = self.custom_data.copy()
                print(f"Loaded custom dataset with {len(self.data)} rows")
                return True

            print(f"Loading data from: {self.data_path}")
            self.data = pd.read_parquet(self.data_path)
            print(f"Loaded {len(self.data)} records")
            return True
        except Exception as exc:
            print(f"Error loading data: {exc}")
            return False

    def start_replay(self, shuffle: bool = False, loop: bool = True):
        if self.data is None:
            if not self.load_data():
                return

        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.is_playing = True
        self.current_index = 0

        self.play_thread = threading.Thread(
            target=self._replay_loop,
            args=(loop,),
            daemon=True
        )
        self.play_thread.start()
        print("Traffic replay started")

    def _replay_loop(self, loop: bool):
        """Internal replay loop"""
        while self.is_playing:
            if self.current_index >= len(self.data):
                if loop:
                    self.current_index = 0
                    print("Replay looped to beginning")
                else:
                    print("Replay finished")
                    self.is_playing = False
                    break

            # Get current packet
            packet = self.data.iloc[self.current_index]

            # Add to queue
            packet_dict = packet.to_dict()
            packet_dict = self._ensure_network_fields(packet_dict)
            packet_dict['timestamp'] = datetime.now()
            packet_dict['replay_index'] = self.current_index

            self.packet_queue.put(packet_dict)

            # Log attacks
            if packet.get('label', 0) == 1:
                attack_info = {
                    'timestamp': datetime.now(),
                    'attack_type': packet.get('attack_cat', 'Unknown'),
                    'src_ip': packet_dict.get('srcip', 'N/A'),
                    'dst_ip': packet_dict.get('dstip', 'N/A'),
                    'service': packet_dict.get('service', '-')
                }
                self.attack_log.append(attack_info)

            self.current_index += 1

            # Sleep to simulate real-time (adjust with speed multiplier)
            time.sleep(0.1 / self.speed_multiplier)

    def stop_replay(self):
        """Stop replay"""
        self.is_playing = False
        if self.play_thread:
            self.play_thread.join(timeout=2)
        print("Traffic replay stopped")

    def get_next_packet(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next packet from replay queue

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Packet dictionary or None if timeout
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_recent_attacks(self, n: int = 10) -> List[Dict]:
        """Get N most recent attacks"""
        return self.attack_log[-n:] if len(self.attack_log) >= n else self.attack_log

    def get_statistics(self) -> Dict:
        """Get replay statistics"""
        if self.data is None:
            return {}

        total = len(self.data)
        attacks = (self.data['label'] == 1).sum()
        normal = (self.data['label'] == 0).sum()

        attack_types = self.data[self.data['label'] == 1]['attack_cat'].value_counts().to_dict()

        return {
            'total_packets': total,
            'normal_traffic': int(normal),
            'attack_traffic': int(attacks),
            'attack_percentage': (attacks / total * 100) if total > 0 else 0,
            'attack_types': attack_types,
            'current_position': self.current_index,
            'progress_percentage': (self.current_index / total * 100) if total > 0 else 0,
            'total_attacks_detected': len(self.attack_log)
        }

    def reset(self):
        """Reset replay to beginning"""
        self.current_index = 0
        self.attack_log.clear()
        self._source_idx = 0
        self._target_idx = 0
        while not self.packet_queue.empty():
            self.packet_queue.get()

    # ------------------------------------------------------------------
    def _ensure_network_fields(self, packet: Dict) -> Dict:
        packet = dict(packet)
        packet['srcip'] = packet.get('srcip') if self._is_valid_ip(packet.get('srcip')) else self._next_source_ip()
        packet['dstip'] = packet.get('dstip') if self._is_valid_ip(packet.get('dstip')) else self._next_target_ip()
        if not packet.get('service'):
            packet['service'] = '-'
        return packet

    @staticmethod
    def _is_valid_ip(value: Optional[str]) -> bool:
        if value is None:
            return False
        value_str = str(value).strip()
        if not value_str or value_str.upper() in {'N/A', 'NA'}:
            return False
        return True

    def _next_source_ip(self) -> str:
        ip = self.source_pool[self._source_idx % len(self.source_pool)]
        self._source_idx += 1
        return ip

    def _next_target_ip(self) -> str:
        ip = self.target_pool[self._target_idx % len(self.target_pool)]
        self._target_idx += 1
        return ip


if __name__ == "__main__":
    # Test traffic replay
    from src.config import DATA_RAW

    replay = TrafficReplay(DATA_RAW / "testing-set.parquet", speed_multiplier=10.0)
    replay.start_replay(shuffle=True)

    print("\nReplaying traffic for 10 seconds...")
    time.sleep(10)

    # Get statistics
    stats = replay.get_statistics()
    print("\n=== Replay Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Get recent attacks
    print("\n=== Recent Attacks ===")
    attacks = replay.get_recent_attacks(5)
    for i, attack in enumerate(attacks, 1):
        print(f"{i}. {attack['attack_type']} from {attack['src_ip']} at {attack['timestamp']}")

    replay.stop_replay()
