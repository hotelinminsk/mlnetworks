"""
Real-Time Monitoring Service
SOLID: Single Responsibility - Monitoring verisi simülasyonu
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random


class MonitoringService:
    """Real-time monitoring verisi simülasyonu için service"""
    
    def __init__(self, baseline_normal: float = 100.0, baseline_std: float = 20.0):
        """
        Args:
            baseline_normal: Normal trafik baseline değeri
            baseline_std: Normal trafik standart sapması
        """
        self.baseline_normal = baseline_normal
        self.baseline_std = baseline_std
        self._history: List[Dict] = []
    
    def generate_traffic_data(
        self,
        n_points: int = 100,
        attack_probability: float = 0.15,
        start_time: datetime = None
    ) -> Dict:
        """
        Trafik verisi simüle et
        
        Args:
            n_points: Zaman noktası sayısı
            attack_probability: Saldırı olasılığı
            start_time: Başlangıç zamanı
            
        Returns:
            Dict: timestamps, total, attacks, normal
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(minutes=n_points)
        
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
        
        # Normal trafik baseline
        normal_traffic = np.random.normal(self.baseline_normal, self.baseline_std, n_points)
        normal_traffic = np.maximum(normal_traffic, 0)  # Negatif değerleri 0 yap
        
        # Saldırı tespitleri
        attack_spikes = np.random.choice([0, 1], n_points, p=[1-attack_probability, attack_probability])
        
        # Saldırı trafiği (daha yüksek değerler)
        attack_traffic = attack_spikes * np.random.normal(300, 50, n_points)
        attack_traffic = np.maximum(attack_traffic, 0)
        
        # Toplam trafik
        total_traffic = normal_traffic + attack_traffic
        
        return {
            'timestamps': timestamps,
            'total': total_traffic,
            'normal': normal_traffic,
            'attacks': attack_spikes.astype(int),
            'attack_traffic': attack_traffic
        }
    
    def generate_live_update(
        self,
        last_timestamp: datetime,
        attack_probability: float = 0.15
    ) -> Dict:
        """
        Tek bir zaman noktası için canlı güncelleme
        
        Args:
            last_timestamp: Son zaman noktası
            attack_probability: Saldırı olasılığı
            
        Returns:
            Dict: Yeni veri noktası
        """
        new_timestamp = last_timestamp + timedelta(minutes=1)
        
        # Normal trafik
        normal = max(0, np.random.normal(self.baseline_normal, self.baseline_std))
        
        # Saldırı kontrolü
        is_attack = random.random() < attack_probability
        attack_traffic = 0
        if is_attack:
            attack_traffic = max(0, np.random.normal(300, 50))
        
        total = normal + attack_traffic
        
        return {
            'timestamp': new_timestamp,
            'total': total,
            'normal': normal,
            'attack': int(is_attack),
            'attack_traffic': attack_traffic
        }
    
    def calculate_metrics(self, data: Dict) -> Dict[str, float]:
        """
        Monitoring metriklerini hesapla
        
        Returns:
            Dict: Metrikler
        """
        total_traffic = data['total'].sum()
        attack_count = int(data['attacks'].sum())
        attack_rate = (attack_count / len(data['attacks'])) * 100 if len(data['attacks']) > 0 else 0
        
        # Ortalama latency (simüle)
        avg_latency = np.random.uniform(2, 8)
        
        # Detection rate (simüle - gerçekte model tahmininden gelir)
        detection_rate = np.random.uniform(95, 99.5)
        
        return {
            'total_traffic': total_traffic,
            'attack_count': attack_count,
            'attack_rate': attack_rate,
            'avg_latency': avg_latency,
            'detection_rate': detection_rate
        }
    
    def get_threat_level(self, attack_count: int, total_points: int) -> Tuple[str, int]:
        """
        Tehdit seviyesini hesapla
        
        Returns:
            Tuple: (level_name, percentage)
        """
        threat_percentage = min(100, (attack_count / total_points) * 100 * 8)
        
        if threat_percentage < 30:
            return "Low", int(threat_percentage)
        elif threat_percentage < 70:
            return "Medium", int(threat_percentage)
        else:
            return "High", int(threat_percentage)

