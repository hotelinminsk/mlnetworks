"""
Professional Alert and Logging System
Real-time intrusion detection alerts with severity levels
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"  # Critical attacks (Backdoor, Shellcode)
    HIGH = "high"          # High-risk attacks (Exploits, DoS)
    MEDIUM = "medium"      # Medium-risk attacks (Reconnaissance, Fuzzers)
    LOW = "low"            # Low-risk attacks (Generic, Analysis)
    INFO = "info"          # Informational


class AlertSystem:
    """
    Professional alert and logging system for IDS
    """

    # Attack severity mapping
    ATTACK_SEVERITY_MAP = {
        'Backdoor': AlertSeverity.CRITICAL,
        'Shellcode': AlertSeverity.CRITICAL,
        'Exploits': AlertSeverity.HIGH,
        'DoS': AlertSeverity.HIGH,
        'Worms': AlertSeverity.HIGH,
        'Reconnaissance': AlertSeverity.MEDIUM,
        'Fuzzers': AlertSeverity.MEDIUM,
        'Analysis': AlertSeverity.LOW,
        'Generic': AlertSeverity.LOW,
    }

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize alert system

        Args:
            log_file: Path to log file (None = no file logging)
        """
        self.alerts = []
        self.log_file = Path(log_file) if log_file else None
        self.alert_count = {severity: 0 for severity in AlertSeverity}

    def create_alert(
        self,
        attack_type: str,
        probability: float,
        source_ip: str,
        dest_ip: str,
        service: str,
        additional_info: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new alert

        Args:
            attack_type: Type of attack detected
            probability: Attack probability (0-1)
            source_ip: Source IP address
            dest_ip: Destination IP address
            service: Service/port targeted
            additional_info: Additional metadata

        Returns:
            Alert dictionary
        """
        severity = self.ATTACK_SEVERITY_MAP.get(attack_type, AlertSeverity.MEDIUM)

        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': datetime.now(),
            'severity': severity.value,
            'attack_type': attack_type,
            'probability': round(probability, 4),
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'service': service,
            'message': self._generate_message(attack_type, source_ip, dest_ip, service),
            'additional_info': additional_info or {}
        }

        self.alerts.append(alert)
        self.alert_count[severity] += 1

        # Log to file if configured
        if self.log_file:
            self._log_to_file(alert)

        return alert

    def _generate_message(self, attack_type: str, src: str, dst: str, service: str) -> str:
        """Generate human-readable alert message"""
        messages = {
            'Backdoor': f"CRITICAL: Backdoor attack detected from {src} targeting {dst}:{service}",
            'Shellcode': f"CRITICAL: Shellcode injection attempt from {src} to {dst}:{service}",
            'Exploits': f"HIGH: Exploit attempt detected - {src} → {dst}:{service}",
            'DoS': f"HIGH: Denial of Service attack from {src} targeting {dst}:{service}",
            'Worms': f"HIGH: Worm activity detected from {src}",
            'Reconnaissance': f"MEDIUM: Reconnaissance scan from {src} targeting {dst}:{service}",
            'Fuzzers': f"MEDIUM: Fuzzing attack from {src} on {dst}:{service}",
            'Analysis': f"LOW: Analysis activity from {src} to {dst}",
            'Generic': f"LOW: Generic malicious activity from {src} to {dst}:{service}"
        }

        return messages.get(attack_type, f"Attack detected: {attack_type} from {src} to {dst}")

    def _log_to_file(self, alert: Dict):
        """Log alert to file"""
        try:
            log_entry = {
                **alert,
                'timestamp': alert['timestamp'].isoformat()
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            print(f"Error logging to file: {e}")

    def get_recent_alerts(self, n: int = 10, severity: Optional[AlertSeverity] = None) -> List[Dict]:
        """
        Get recent alerts

        Args:
            n: Number of alerts to return
            severity: Filter by severity (None = all)

        Returns:
            List of alerts
        """
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [a for a in self.alerts if a['severity'] == severity.value]

        return filtered_alerts[-n:] if len(filtered_alerts) >= n else filtered_alerts

    def get_critical_alerts(self, n: int = 5) -> List[Dict]:
        """Get recent critical alerts"""
        return self.get_recent_alerts(n, AlertSeverity.CRITICAL)

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        if not self.alerts:
            return {
                'total': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            }

        return {
            'total': len(self.alerts),
            'critical': self.alert_count[AlertSeverity.CRITICAL],
            'high': self.alert_count[AlertSeverity.HIGH],
            'medium': self.alert_count[AlertSeverity.MEDIUM],
            'low': self.alert_count[AlertSeverity.LOW],
            'info': self.alert_count[AlertSeverity.INFO]
        }

    def get_attack_distribution(self) -> pd.DataFrame:
        """Get distribution of attack types"""
        if not self.alerts:
            return pd.DataFrame()

        attack_types = [a['attack_type'] for a in self.alerts]
        dist = pd.Series(attack_types).value_counts()
        return dist.to_frame('count')

    def get_top_attackers(self, n: int = 10) -> pd.DataFrame:
        """Get top N attacking IP addresses"""
        if not self.alerts:
            return pd.DataFrame()

        attackers = [a['source_ip'] for a in self.alerts]
        top = pd.Series(attackers).value_counts().head(n)
        return top.to_frame('attacks')

    def get_targeted_services(self) -> pd.DataFrame:
        """Get most targeted services"""
        if not self.alerts:
            return pd.DataFrame()

        services = [a['service'] for a in self.alerts if a['service'] != '-']
        dist = pd.Series(services).value_counts()
        return dist.to_frame('attacks')

    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        self.alert_count = {severity: 0 for severity in AlertSeverity}

    def export_alerts(self, output_file: str, format: str = 'json'):
        """
        Export alerts to file

        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_file, 'w') as f:
                alerts_json = [
                    {**a, 'timestamp': a['timestamp'].isoformat()}
                    for a in self.alerts
                ]
                json.dump(alerts_json, f, indent=2)

        elif format == 'csv':
            df = pd.DataFrame(self.alerts)
            df.to_csv(output_file, index=False)

        print(f"Exported {len(self.alerts)} alerts to {output_file}")


if __name__ == "__main__":
    # Test alert system
    alert_system = AlertSystem(log_file="alerts.log")

    # Create test alerts
    alert_system.create_alert(
        attack_type="Backdoor",
        probability=0.95,
        source_ip="192.168.1.100",
        dest_ip="10.0.0.50",
        service="ssh"
    )

    alert_system.create_alert(
        attack_type="DoS",
        probability=0.88,
        source_ip="192.168.1.200",
        dest_ip="10.0.0.80",
        service="http"
    )

    alert_system.create_alert(
        attack_type="Reconnaissance",
        probability=0.72,
        source_ip="192.168.1.150",
        dest_ip="10.0.0.1",
        service="-"
    )

    print("=== Alert Summary ===")
    print(alert_system.get_alert_summary())

    print("\n=== Recent Alerts ===")
    for alert in alert_system.get_recent_alerts(10):
        print(f"[{alert['severity'].upper()}] {alert['message']}")

    print("\n=== Attack Distribution ===")
    print(alert_system.get_attack_distribution())
