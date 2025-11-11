"""
Real-time IDS Service
Combines network capture, replay, prediction, and alerting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.traffic_replay import TrafficReplay
from src.alert_system import AlertSystem
from src.config import DATA_RAW, MODELS
from joblib import load


class RealtimeIDSService:
    """
    Professional Real-time Intrusion Detection Service
    - Replays real UNSW-NB15 data
    - Makes predictions with trained models
    - Generates professional alerts
    - Tracks statistics
    """

    def __init__(self, model_name: str = "gradient_boosting"):
        """
        Initialize IDS service

        Args:
            model_name: Name of ML model to use
        """
        # Load model and preprocessor
        self.model = load(MODELS / f"{model_name}.joblib")
        self.preprocessor = load(MODELS / "preprocess_ct.joblib")
        self.model_name = model_name

        # Initialize components
        self.replay = TrafficReplay(
            DATA_RAW / "testing-set.parquet",
            speed_multiplier=10.0  # 10x speed for demo
        )
        self.alert_system = AlertSystem()

        # Statistics
        self.stats = {
            'packets_processed': 0,
            'attacks_detected': 0,
            'normal_traffic': 0,
            'false_positives': 0,
            'true_positives': 0
        }

        self.recent_packets = []
        self.max_recent = 100

    def start(self, shuffle: bool = True):
        """Start the IDS service"""
        self.replay.start_replay(shuffle=shuffle, loop=True)
        print(f"IDS Service started with model: {self.model_name}")

    def stop(self):
        """Stop the IDS service"""
        self.replay.stop_replay()
        print("IDS Service stopped")

    def process_next_packet(self) -> Optional[Dict]:
        """
        Process next packet from replay

        Returns:
            Packet info with prediction and alert if attack detected
        """
        packet = self.replay.get_next_packet(timeout=0.5)
        if not packet:
            return None

        # Prepare features for prediction
        features_df = pd.DataFrame([packet])

        # Remove labels if present
        if 'label' in features_df.columns:
            true_label = packet['label']
            attack_cat = packet.get('attack_cat', 'Unknown')
            features_df = features_df.drop(columns=['label', 'attack_cat'], errors='ignore')
        else:
            true_label = None
            attack_cat = None

        # Make prediction
        try:
            X = self.preprocessor.transform(features_df)

            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X)[0][1]
            else:
                score = self.model.decision_function(X)[0]
                prob = 1 / (1 + np.exp(-score))

            # Use threshold
            threshold = 0.7
            prediction = 1 if prob >= threshold else 0

        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = 0
            prob = 0.0

        # Update statistics
        self.stats['packets_processed'] += 1

        if prediction == 1:
            self.stats['attacks_detected'] += 1

            # Check if it's true or false positive
            if true_label is not None:
                if true_label == 1:
                    self.stats['true_positives'] += 1
                else:
                    self.stats['false_positives'] += 1

            # Create alert
            alert = self.alert_system.create_alert(
                attack_type=attack_cat if attack_cat and attack_cat != 'Normal' else 'Generic',
                probability=prob,
                source_ip=packet.get('srcip', 'N/A'),
                dest_ip=packet.get('dstip', 'N/A'),
                service=packet.get('service', '-'),
                additional_info={
                    'model': self.model_name,
                    'true_label': true_label,
                    'protocol': packet.get('proto', 0)
                }
            )
        else:
            self.stats['normal_traffic'] += 1
            alert = None

        # Package result
        result = {
            'timestamp': packet.get('timestamp'),
            'srcip': packet.get('srcip', 'N/A'),
            'dstip': packet.get('dstip', 'N/A'),
            'service': packet.get('service', '-'),
            'prediction': 'attack' if prediction == 1 else 'normal',
            'probability': round(prob, 4),
            'true_label': true_label,
            'attack_type': attack_cat if attack_cat else None,
            'alert': alert
        }

        # Add to recent packets
        self.recent_packets.append(result)
        if len(self.recent_packets) > self.max_recent:
            self.recent_packets.pop(0)

        return result

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        stats = {
            **self.stats,
            'alert_summary': self.alert_system.get_alert_summary(),
            'replay_stats': self.replay.get_statistics(),
            'accuracy': self._calculate_accuracy()
        }
        return stats

    def _calculate_accuracy(self) -> float:
        """Calculate accuracy based on true/false positives"""
        total = self.stats['true_positives'] + self.stats['false_positives']
        if total == 0:
            return 0.0
        return (self.stats['true_positives'] / total) * 100

    def get_recent_packets(self, n: int = 20) -> List[Dict]:
        """Get N most recent packets"""
        return self.recent_packets[-n:] if len(self.recent_packets) >= n else self.recent_packets

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get N most recent alerts"""
        return self.alert_system.get_recent_alerts(n)

    def get_critical_alerts(self) -> List[Dict]:
        """Get recent critical alerts"""
        return self.alert_system.get_critical_alerts()

    def get_attack_distribution(self) -> pd.DataFrame:
        """Get distribution of detected attacks"""
        return self.alert_system.get_attack_distribution()

    def reset(self):
        """Reset all statistics and alerts"""
        self.stats = {
            'packets_processed': 0,
            'attacks_detected': 0,
            'normal_traffic': 0,
            'false_positives': 0,
            'true_positives': 0
        }
        self.recent_packets.clear()
        self.alert_system.clear_alerts()
        self.replay.reset()
