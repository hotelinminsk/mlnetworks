"""
Real-Time Monitoring Service
SOLID Principles:
- Single Responsibility: Sadece real-time monitoring verisi yönetimi
- Open/Closed: Genişletmeye açık, değişikliğe kapalı
- Dependency Inversion: Interface'lere bağımlı
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import streamlit as st


class RealtimeMonitoringService:
    """Real-time monitoring verisi yönetimi için service sınıfı"""
    
    def __init__(self, max_points: int = 100):
        """
        Args:
            max_points: Kayan pencerede tutulacak maksimum nokta sayısı
        """
        self.max_points = max_points
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Session state'i başlat"""
        if 'monitoring_data' not in st.session_state:
            st.session_state['monitoring_data'] = pd.DataFrame({
                'timestamp': [],
                'total_traffic': [],
                'normal_traffic': [],
                'attack_traffic': [],
                'is_attack': []
            })
            st.session_state['monitoring_active'] = True
            st.session_state['update_counter'] = 0
    
    def generate_data_point(self) -> Optional[pd.DataFrame]:
        """
        Yeni veri noktası üret
        
        Returns:
            Yeni veri noktası içeren DataFrame veya None (pause durumunda)
        """
        if not st.session_state.get('monitoring_active', True):
            return None
        
        current_time = datetime.now()
        
        # Normal trafik (Gaussian distribution)
        normal_traffic = max(0, np.random.normal(100, 20))
        
        # Saldırı kontrolü (15% olasılık)
        is_attack = np.random.random() < 0.15
        attack_traffic = max(0, np.random.normal(300, 50)) if is_attack else 0
        
        total_traffic = normal_traffic + attack_traffic
        
        # Yeni veri noktası
        new_row = pd.DataFrame({
            'timestamp': [current_time],
            'total_traffic': [total_traffic],
            'normal_traffic': [normal_traffic],
            'attack_traffic': [attack_traffic],
            'is_attack': [int(is_attack)]
        })
        
        return new_row
    
    def update_data(self, new_row: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi güncelle (kayan pencere mantığı)
        
        Args:
            new_row: Eklenecek yeni veri noktası
            
        Returns:
            Güncellenmiş DataFrame
        """
        monitoring_df = st.session_state['monitoring_data']
        
        # Yeni veriyi ekle
        monitoring_df = pd.concat([monitoring_df, new_row], ignore_index=True)
        
        # Kayan pencere: Son max_points noktayı tut
        if len(monitoring_df) > self.max_points:
            monitoring_df = monitoring_df.tail(self.max_points).reset_index(drop=True)
        
        # Session state'i güncelle
        st.session_state['monitoring_data'] = monitoring_df
        st.session_state['update_counter'] = st.session_state.get('update_counter', 0) + 1
        
        return monitoring_df
    
    def get_current_data(self) -> pd.DataFrame:
        """Mevcut monitoring verisini getir"""
        return st.session_state.get('monitoring_data', pd.DataFrame())
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Monitoring metriklerini hesapla
        
        Args:
            df: Monitoring DataFrame
            
        Returns:
            Metrikler dictionary
        """
        if len(df) == 0:
            return {
                'total_traffic': 0,
                'attack_count': 0,
                'attack_rate': 0.0,
                'avg_traffic': 0.0
            }
        
        return {
            'total_traffic': df['total_traffic'].sum(),
            'attack_count': int(df['is_attack'].sum()),
            'attack_rate': (df['is_attack'].sum() / len(df)) * 100,
            'avg_traffic': df['total_traffic'].mean()
        }
    
    def reset(self) -> None:
        """Monitoring verisini sıfırla"""
        st.session_state['monitoring_data'] = pd.DataFrame({
            'timestamp': [],
            'total_traffic': [],
            'normal_traffic': [],
            'attack_traffic': [],
            'is_attack': []
        })
        st.session_state['update_counter'] = 0
    
    def pause(self) -> None:
        """Monitoring'i duraklat"""
        st.session_state['monitoring_active'] = False
    
    def resume(self) -> None:
        """Monitoring'i devam ettir"""
        st.session_state['monitoring_active'] = True
    
    def is_active(self) -> bool:
        """Monitoring aktif mi?"""
        return st.session_state.get('monitoring_active', True)

