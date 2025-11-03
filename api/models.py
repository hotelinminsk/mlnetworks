"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TrafficFeatures(BaseModel):
    """Network traffic features for intrusion detection"""

    # Duration and basic stats
    dur: float = Field(..., description="Connection duration in seconds")
    spkts: int = Field(..., description="Source to destination packet count")
    dpkts: int = Field(..., description="Destination to source packet count")
    sbytes: int = Field(..., description="Source to destination bytes")
    dbytes: int = Field(..., description="Destination to source bytes")

    # Rates and loads
    rate: float = Field(..., description="Packet rate")
    sload: float = Field(..., description="Source bits per second")
    dload: float = Field(..., description="Destination bits per second")

    # Protocol info
    proto: str = Field(..., description="Protocol (tcp, udp, etc.)")
    service: str = Field(..., description="Service type or '-'")
    state: str = Field(..., description="Connection state")

    # Loss and timing
    sloss: int = Field(default=0, description="Source packets retransmitted")
    dloss: int = Field(default=0, description="Destination packets retransmitted")
    sinpkt: float = Field(default=0.0, description="Source inter-packet time")
    dinpkt: float = Field(default=0.0, description="Destination inter-packet time")
    sjit: float = Field(default=0.0, description="Source jitter")
    djit: float = Field(default=0.0, description="Destination jitter")

    # TCP specific
    swin: int = Field(default=0, description="Source TCP window")
    dwin: int = Field(default=0, description="Destination TCP window")
    stcpb: int = Field(default=0, description="Source TCP base sequence number")
    dtcpb: int = Field(default=0, description="Destination TCP base sequence number")
    tcprtt: float = Field(default=0.0, description="TCP round trip time")
    synack: float = Field(default=0.0, description="SYN to ACK time")
    ackdat: float = Field(default=0.0, description="ACK to data time")

    # Means
    smean: int = Field(default=0, description="Mean of flow packet size (source)")
    dmean: int = Field(default=0, description="Mean of flow packet size (destination)")

    # HTTP specific
    trans_depth: int = Field(default=0, description="HTTP pipeline depth")
    response_body_len: int = Field(default=0, description="HTTP response body length")

    # Connection tracking
    ct_src_dport_ltm: int = Field(default=1, description="Connections with same source-dest port")
    ct_dst_sport_ltm: int = Field(default=1, description="Connections with same dest-source port")

    # FTP specific
    is_ftp_login: int = Field(default=0, description="FTP login detected (0/1)")
    ct_ftp_cmd: int = Field(default=0, description="FTP command count")

    # HTTP methods
    ct_flw_http_mthd: int = Field(default=0, description="HTTP methods in flow")

    # Flags
    is_sm_ips_ports: int = Field(default=0, description="Same IPs/ports (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "dur": 0.5,
                "spkts": 20,
                "dpkts": 15,
                "sbytes": 2048,
                "dbytes": 1536,
                "rate": 70.0,
                "sload": 32768.0,
                "dload": 24576.0,
                "proto": "tcp",
                "service": "http",
                "state": "FIN",
                "sloss": 0,
                "dloss": 0,
                "sinpkt": 0.025,
                "dinpkt": 0.033,
                "sjit": 0.001,
                "djit": 0.002,
                "swin": 255,
                "dwin": 255,
                "stcpb": 1000,
                "dtcpb": 2000,
                "tcprtt": 0.01,
                "synack": 0.005,
                "ackdat": 0.003,
                "smean": 102,
                "dmean": 102,
                "trans_depth": 0,
                "response_body_len": 0,
                "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1,
                "is_ftp_login": 0,
                "ct_ftp_cmd": 0,
                "ct_flw_http_mthd": 1,
                "is_sm_ips_ports": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response from prediction endpoint"""

    prediction: str = Field(..., description="Predicted class: 'normal' or 'attack'")
    probability: float = Field(..., description="Probability of being an attack (0-1)")
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', or 'high'")
    model_used: str = Field(..., description="Model name used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "attack",
                "probability": 0.95,
                "confidence": "high",
                "model_used": "gradient_boosting",
                "timestamp": "2024-11-02T15:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""

    samples: List[TrafficFeatures] = Field(..., description="List of traffic samples to predict")

    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    {
                        "dur": 0.5,
                        "spkts": 20,
                        "dpkts": 15,
                        "sbytes": 2048,
                        "dbytes": 1536,
                        "rate": 70.0,
                        "sload": 32768.0,
                        "dload": 24576.0,
                        "proto": "tcp",
                        "service": "http",
                        "state": "FIN"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint"""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_samples: int = Field(..., description="Total number of samples processed")
    attacks_detected: int = Field(..., description="Number of attacks detected")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Currently loaded model name")
    version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


class ModelInfo(BaseModel):
    """Model information"""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type/algorithm")
    roc_auc: float = Field(..., description="ROC AUC score")
    accuracy: float = Field(..., description="Accuracy score")
    f1_score: float = Field(..., description="F1 score")
    trained_date: Optional[str] = Field(None, description="Training date")
