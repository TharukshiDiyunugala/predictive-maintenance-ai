"""
Alert Management System
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List
from datetime import datetime
import requests
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manage and send alerts for equipment failures
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.channels = config.get('channels', ['email'])
        self.thresholds = config.get('thresholds', {
            'critical': 0.9,
            'warning': 0.7,
            'info': 0.5
        })
    
    def should_alert(self, failure_prob: float, severity: str) -> bool:
        """
        Determine if alert should be sent
        
        Args:
            failure_prob: Failure probability
            severity: Severity level
        
        Returns:
            True if alert should be sent
        """
        severity_lower = severity.lower()
        threshold = self.thresholds.get(severity_lower, 0.5)
        return failure_prob >= threshold
    
    def send_alert(
        self,
        equipment_id: str,
        prediction: Dict[str, Any],
        alert_type: str = 'failure_prediction'
    ):
        """
        Send alert through configured channels
        
        Args:
            equipment_id: Equipment identifier
            prediction: Prediction results
            alert_type: Type of alert
        """
        severity = prediction.get('severity', 'MEDIUM')
        failure_prob = prediction.get('failure_prob', 0)
        
        if not self.should_alert(failure_prob, severity):
            logger.info(f"Alert threshold not met for {equipment_id}")
            return
        
        message = self._create_alert_message(equipment_id, prediction, alert_type)
        
        # Send through each channel
        for channel in self.channels:
            try:
                if channel == 'email':
                    self._send_email_alert(message)
                elif channel == 'slack':
                    self._send_slack_alert(message)
                elif channel == 'webhook':
                    self._send_webhook_alert(message)
                
                logger.info(f"Alert sent via {channel} for {equipment_id}")
            
            except Exception as e:
                logger.error(f"Error sending alert via {channel}: {e}")
    
    def _create_alert_message(
        self,
        equipment_id: str,
        prediction: Dict[str, Any],
        alert_type: str
    ) -> Dict[str, Any]:
        """Create alert message"""
        return {
            'alert_type': alert_type,
            'equipment_id': equipment_id,
            'severity': prediction.get('severity', 'UNKNOWN'),
            'failure_probability': prediction.get('failure_prob', 0),
            'days_to_failure': prediction.get('days_to_failure'),
            'recommendations': prediction.get('recommendations', []),
            'timestamp': datetime.now().isoformat(),
            'subject': f"[{prediction.get('severity')}] Equipment {equipment_id} - Failure Alert",
            'body': self._format_alert_body(equipment_id, prediction)
        }
    
    def _format_alert_body(self, equipment_id: str, prediction: Dict[str, Any]) -> str:
        """Format alert message body"""
        body = f"""
Equipment Failure Alert
=====================

Equipment ID: {equipment_id}
Severity: {prediction.get('severity', 'UNKNOWN')}
Failure Probability: {prediction.get('failure_prob', 0):.2%}
Days to Failure: {prediction.get('days_to_failure', 'Unknown')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Recommendations:
"""
        for i, rec in enumerate(prediction.get('recommendations', []), 1):
            body += f"{i}. {rec}\n"
        
        body += "\nPlease take immediate action to prevent equipment failure.\n"
        
        return body
    
    def _send_email_alert(self, message: Dict[str, Any]):
        """Send email alert"""
        email_config = self.config.get('email', {})
        
        if not email_config:
            logger.warning("Email configuration not found")
            return
        
        msg = MIMEMultipart()
        msg['From'] = email_config.get('from_address')
        msg['To'] = ', '.join(email_config.get('to_addresses', []))
        msg['Subject'] = message['subject']
        
        msg.attach(MIMEText(message['body'], 'plain'))
        
        try:
            server = smtplib.SMTP(
                email_config.get('smtp_server'),
                email_config.get('smtp_port', 587)
            )
            server.starttls()
            
            # If credentials are provided
            if email_config.get('username') and email_config.get('password'):
                server.login(
                    email_config.get('username'),
                    email_config.get('password')
                )
            
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent successfully")
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    def _send_slack_alert(self, message: Dict[str, Any]):
        """Send Slack alert"""
        slack_config = self.config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        payload = {
            'text': message['subject'],
            'attachments': [{
                'color': self._get_severity_color(message['severity']),
                'text': message['body'],
                'footer': 'Predictive Maintenance System',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        
        logger.info("Slack alert sent successfully")
    
    def _send_webhook_alert(self, message: Dict[str, Any]):
        """Send webhook alert"""
        webhook_config = self.config.get('webhook', {})
        url = webhook_config.get('url')
        method = webhook_config.get('method', 'POST')
        
        if not url:
            logger.warning("Webhook URL not configured")
            return
        
        if method.upper() == 'POST':
            response = requests.post(url, json=message)
        else:
            response = requests.get(url, params=message)
        
        response.raise_for_status()
        
        logger.info("Webhook alert sent successfully")
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color code for severity"""
        colors = {
            'CRITICAL': 'danger',
            'HIGH': 'warning',
            'MEDIUM': '#ffcc00',
            'LOW': 'good'
        }
        return colors.get(severity, 'good')
