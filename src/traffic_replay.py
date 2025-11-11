"""
Traffic Replay System
Replays UNSW-NB15 data as if it's live network traffic
Perfect for professional demonstrations
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import threading
import queue


class TrafficReplay:
    """
    Replays real UNSW-NB15 data as live traffic
    Makes the demo look professional with real attack scenarios
    """

    def __init__(self, data_path: str, speed_multiplier: float = 1.0):
        """
        Initialize traffic replay

        Args:
            data_path: Path to UNSW-NB15 data file
            speed_multiplier: Replay speed (1.0 = realtime, 2.0 = 2x speed, 0.5 = slow motion)
        """
        self.data_path = Path(data_path)
        self.speed_multiplier = speed_multiplier
        self.data = None
        self.current_index = 0
        self.is_playing = False
        self.play_thread = None
        self.packet_queue = queue.Queue()
        self.attack_log = []

    def load_data(self) -> bool:
        """Load UNSW-NB15 data"""
        try:
            print(f"Loading data from: {self.data_path}")
            self.data = pd.read_parquet(self.data_path)
            print(f"Loaded {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def start_replay(self, shuffle: bool = False, loop: bool = True):
        """
        Start replaying traffic

        Args:
            shuffle: Shuffle data before replay for variety
            loop: Loop replay when reaching end
        """
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
            packet_dict['timestamp'] = datetime.now()
            packet_dict['replay_index'] = self.current_index

            self.packet_queue.put(packet_dict)

            # Log attacks
            if packet.get('label', 0) == 1:
                attack_info = {
                    'timestamp': datetime.now(),
                    'attack_type': packet.get('attack_cat', 'Unknown'),
                    'src_ip': packet.get('srcip', 'N/A'),
                    'dst_ip': packet.get('dstip', 'N/A'),
                    'service': packet.get('service', 'N/A')
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
        while not self.packet_queue.empty():
            self.packet_queue.get()


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
