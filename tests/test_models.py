"""
Unit tests for database models
Tests model validation, relationships, and serialization
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.model import Model, ModelDeployment
from app.models.monitoring import (
    PredictionLogDB, ModelPerformanceMetricsDB, SystemHealthMetricDB,
    AlertDB, AuditLogDB
)


class TestModelClass:
    """Test Model database model"""
    
    def test_model_creation(self, test_session):
        """Test basic model creation"""
        model_data = {
            "name": "test_model",
            "description": "A test model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0",
            "file_name": "test_model.joblib",
            "file_size": 1024,
            "file_hash": "abcd1234efgh5678"
        }
        
        model = Model(**model_data)
        test_session.add(model)
        test_session.commit()
        
        assert model.id is not None
        assert model.name == "test_model"
        assert model.model_type == "classification"
        assert model.framework == "sklearn"
        assert model.created_at is not None
        assert model.updated_at is not None
    
    def test_model_validation_required_fields(self, test_session):
        """Test model validation for required fields"""
        # Missing required fields should raise error
        with pytest.raises(Exception):  # Could be IntegrityError or ValidationError
            model = Model()
            test_session.add(model)
            test_session.commit()
    
    def test_model_unique_hash_constraint(self, test_session):
        """Test unique file hash constraint"""
        model_data = {
            "name": "first_model",
            "description": "First model",
            "model_type": "classification",
            "framework": "sklearn",
            "version": "1.0.0",
            "file_name": "test1.joblib",
            "file_size": 1024,
            "file_hash": "duplicate_hash"
        }
        
        model1 = Model(**model_data)
        test_session.add(model1)
        test_session.commit()
        
        # Try to create another model with same hash
        model_data["name"] = "second_model"
        model_data["file_name"] = "test2.joblib"
        model2 = Model(**model_data)
        
        test_session.add(model2)
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_model_json_fields(self, test_session):
        """Test JSON field storage and retrieval"""
        tags = ["ml", "classification", "production"]
        
        model = Model(
            name="json_test_model",
            description="Testing JSON fields",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            file_name="test_model.joblib",
            file_size=1024,
            file_hash="json_test_hash",
            tags=tags
        )
        
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Verify JSON fields are stored and retrieved correctly
        assert model.tags == tags
        assert "ml" in model.tags
        assert "classification" in model.tags
    
    def test_model_relationships(self, test_session):
        """Test model relationships with deployments"""
        model = Model(
            name="relationship_test",
            description="Testing relationships",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            file_name="test_model.joblib",
            file_size=1024,
            file_hash="relationship_test_hash"
        )
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Create deployments for this model
        deployment1 = ModelDeployment(
            deployment_name="deployment1",
            model_id=model.id,
            deployment_url="http://localhost:3001",
            status="pending",
            configuration={},
            replicas=1
        )
        
        deployment2 = ModelDeployment(
            deployment_name="deployment2",
            model_id=model.id,
            deployment_url="http://localhost:3002",
            status="pending",
            configuration={},
            replicas=1
        )
        
        test_session.add_all([deployment1, deployment2])
        test_session.commit()
        
        # Test relationship
        test_session.refresh(model)
        assert len(model.deployments) == 2
        assert deployment1 in model.deployments
        assert deployment2 in model.deployments
    
    def test_model_timestamps(self, test_session):
        """Test model timestamp fields"""
        from datetime import datetime, timezone
        creation_time = datetime.now(timezone.utc)
        
        model = Model(
            name="timestamp_test_model",
            description="Testing timestamps",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            file_name="test_model.joblib",
            file_size=1024,
            file_hash="timestamp_test_hash"
        )
        
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Convert to UTC if not already timezone-aware for comparison
        model_created_at = model.created_at
        if model_created_at.tzinfo is None:
            model_created_at = model_created_at.replace(tzinfo=timezone.utc)
        
        # Timestamps should be populated
        assert model_created_at >= creation_time
        assert model.updated_at is not None
        
        # Should be close to current time (within 1 second)
        time_diff = abs((datetime.now(timezone.utc) - model_created_at).total_seconds())
        assert time_diff < 1.0


class TestDeploymentClass:
    """Test Deployment database model"""
    
    def test_deployment_creation(self, test_session, test_model):
        """Test basic deployment creation"""
        deployment_data = {
            "deployment_name": "test_deployment",
            "model_id": test_model.id,
            "deployment_url": "http://localhost:3001",
            "status": "pending",
            "configuration": {"cpu": "100m", "memory": "256Mi"},
            "cpu_request": 0.1,
            "memory_request": "256Mi",
            "replicas": 1
        }
        
        deployment = ModelDeployment(**deployment_data)
        test_session.add(deployment)
        test_session.commit()
        
        assert deployment.id is not None
        assert deployment.deployment_name == "test_deployment"
        assert deployment.model_id == test_model.id
        assert deployment.deployment_url == "http://localhost:3001"
        assert deployment.status == "pending"
        assert deployment.created_at is not None
    
    def test_deployment_model_relationship(self, test_session, test_model):
        """Test deployment-model relationship"""
        deployment = ModelDeployment(
            deployment_name="relationship_test",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status="pending",
            configuration={},
            replicas=1
        )
        test_session.add(deployment)
        test_session.commit()
        test_session.refresh(deployment)
        
        # Test forward relationship
        assert deployment.model.id == test_model.id
        assert deployment.model.name == test_model.name
        
        # Test reverse relationship
        test_session.refresh(test_model)
        assert deployment in test_model.deployments
    
    def test_deployment_status_values(self, test_session, test_model):
        """Test deployment status values"""
        valid_statuses = ["pending", "deploying", "running", "failed", "stopped"]
        
        for status in valid_statuses:
            deployment = ModelDeployment(
                deployment_name=f"status_test_{status}",
                model_id=test_model.id,
                deployment_url=f"http://localhost:300{valid_statuses.index(status)}",
                status=status,
                configuration={},
                replicas=1
            )
            test_session.add(deployment)
            test_session.commit()
            test_session.refresh(deployment)
            
            assert deployment.status == status
    
    def test_deployment_json_configuration(self, test_session, test_model):
        """Test JSON configuration storage"""
        configuration = {
            "cpu": "500m",
            "memory": "1Gi",
            "gpu": {"type": "nvidia-tesla-k80", "count": 1}
        }
        
        deployment = ModelDeployment(
            deployment_name="json_config_test",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status="pending",
            configuration=configuration,
            cpu_request=0.5,
            memory_request="1Gi",
            replicas=2
        )
        
        test_session.add(deployment)
        test_session.commit()
        test_session.refresh(deployment)
        
        assert deployment.configuration == configuration
        assert deployment.configuration["cpu"] == "500m"
        assert deployment.cpu_request == 0.5
        assert deployment.replicas == 2


class TestPredictionLog:
    """Test PredictionLog monitoring model"""
    
    def test_prediction_log_creation(self, test_session, test_model):
        """Test prediction log creation"""
        log_data = {
            "id": "test_log_123",
            "model_id": test_model.id,
            "request_id": "req_123",
            "input_data": {"feature1": 0.5, "feature2": "test"},
            "output_data": {"class": "A", "probability": 0.85},
            "latency_ms": 150.5,
            "api_endpoint": "/predict",
            "success": True
        }
        
        log = PredictionLogDB(**log_data)
        test_session.add(log)
        test_session.commit()
        
        assert log.id is not None
        assert log.model_id == test_model.id
        assert log.input_data == {"feature1": 0.5, "feature2": "test"}
        assert log.output_data == {"class": "A", "probability": 0.85}
        assert log.latency_ms == 150.5
        assert log.success is True
        assert log.timestamp is not None
    
    def test_prediction_log_success_values(self, test_session, test_model):
        """Test prediction log success values"""
        log = PredictionLogDB(
            id="test_log_success",
            model_id=test_model.id,
            request_id="req_123",
            input_data={"test": "data"},
            output_data={"result": "test"},
            latency_ms=100.0,
            api_endpoint="/predict",
            success=True
        )
        test_session.add(log)
        test_session.commit()
        test_session.refresh(log)
        
        assert log.success is True


class TestModelPerformanceMetric:
    """Test ModelPerformanceMetric monitoring model"""
    
    def test_performance_metric_creation(self, test_session, test_model):
        """Test performance metric creation"""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        
        metric = ModelPerformanceMetricsDB(
            id="test_metric_123",
            model_id=test_model.id,
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            requests_per_minute=10.0,
            avg_latency_ms=150.0,
            p50_latency_ms=140.0,
            p95_latency_ms=200.0,
            p99_latency_ms=250.0,
            max_latency_ms=300.0,
            success_rate=0.95,
            error_rate=0.05
        )
        
        test_session.add(metric)
        test_session.commit()
        
        assert metric.id is not None
        assert metric.model_id == test_model.id
        assert metric.total_requests == 100
        assert metric.success_rate == 0.95
        assert metric.created_at is not None


class TestSystemHealthMetric:
    """Test SystemHealthMetric monitoring model"""
    
    def test_system_health_metric_creation(self, test_session):
        """Test system health metric creation"""
        metric = SystemHealthMetricDB(
            id="test_health_123",
            component="api_server",
            metric_type="cpu_usage",
            value=45.2,
            unit="percentage"
        )
        
        test_session.add(metric)
        test_session.commit()
        
        assert metric.id is not None
        assert metric.component == "api_server"
        assert metric.metric_type == "cpu_usage"
        assert metric.value == 45.2
        assert metric.unit == "percentage"
        assert metric.timestamp is not None
    
    def test_system_health_metric_components(self, test_session):
        """Test different system components"""
        components = [
            ("api_server", "cpu_usage", 45.2),
            ("database", "connection_pool", 80.0),
            ("redis", "memory_usage", 30.5),
            ("nginx", "requests_per_second", 150.0)
        ]
        
        for component, metric_type, value in components:
            metric = SystemHealthMetricDB(
                id=f"test_{component}_{metric_type}",
                component=component,
                metric_type=metric_type,
                value=value,
                unit="percentage"
            )
            test_session.add(metric)
            test_session.commit()
            test_session.refresh(metric)
            
            assert metric.component == component
            assert metric.metric_type == metric_type
            assert metric.value == value


class TestAlert:
    """Test Alert monitoring model"""
    
    def test_alert_creation(self, test_session):
        """Test alert creation"""
        alert = AlertDB(
            id="test_alert_123",
            severity="warning",
            component="api_server",
            title="High CPU Usage",
            description="CPU usage exceeded 70% threshold",
            additional_data={"cpu_usage": 75.5, "threshold": 70.0}
        )
        
        test_session.add(alert)
        test_session.commit()
        
        assert alert.id is not None
        assert alert.severity == "warning"
        assert alert.component == "api_server"
        assert alert.title == "High CPU Usage"
        assert alert.is_active is True
        assert alert.triggered_at is not None
    
    def test_alert_severity_levels(self, test_session):
        """Test alert severity levels"""
        severities = ["info", "warning", "error", "critical"]
        
        for i, severity in enumerate(severities):
            alert = AlertDB(
                id=f"test_alert_{i}",
                severity=severity,
                component="test_component",
                title=f"Test {severity} alert",
                description=f"This is a {severity} level alert"
            )
            test_session.add(alert)
            test_session.commit()
            test_session.refresh(alert)
            
            assert alert.severity == severity
    
    def test_alert_resolution(self, test_session):
        """Test alert resolution"""
        alert = AlertDB(
            id="test_alert_resolution",
            severity="warning",
            component="test",
            title="Test Alert",
            description="Test message"
        )
        
        test_session.add(alert)
        test_session.commit()
        
        # Initially unresolved
        assert alert.is_active is True
        assert alert.resolved_at is None
        
        # Resolve alert
        alert.is_active = False
        alert.resolved_at = datetime.now(timezone.utc)
        test_session.commit()
        test_session.refresh(alert)
        
        assert alert.is_active is False
        assert alert.resolved_at is not None


class TestAuditLog:
    """Test AuditLog monitoring model"""
    
    def test_audit_log_creation(self, test_session):
        """Test audit log creation"""
        audit = AuditLogDB(
            id="test_audit_123",
            action="model_upload",
            resource_type="model",
            resource_id="model_123",
            user_id="user_456",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Test Browser)",
            old_values={"status": "pending"},
            new_values={"status": "active"}
        )
        
        test_session.add(audit)
        test_session.commit()
        
        assert audit.id is not None
        assert audit.action == "model_upload"
        assert audit.resource_type == "model"
        assert audit.resource_id == "model_123"
        assert audit.user_id == "user_456"
        assert audit.ip_address == "192.168.1.100"
        assert audit.timestamp is not None
    
    def test_audit_log_actions(self, test_session):
        """Test different audit log actions"""
        actions = [
            ("model_upload", "model"),
            ("model_delete", "model"),
            ("deployment_create", "deployment"),
            ("deployment_stop", "deployment"),
            ("schema_update", "schema")
        ]
        
        for i, (action, resource_type) in enumerate(actions):
            audit = AuditLogDB(
                id=f"test_audit_{i}",
                action=action,
                resource_type=resource_type,
                resource_id=f"{resource_type}_123",
                user_id="test_user"
            )
            test_session.add(audit)
            test_session.commit()
            test_session.refresh(audit)
            
            assert audit.action == action
            assert audit.resource_type == resource_type


class TestModelIntegration:
    """Integration tests for model relationships and constraints"""
    
    def test_model_deployment_cascade(self, test_session, test_model):
        """Test relationships and data integrity between models"""
        # Create deployment
        deployment = ModelDeployment(
            deployment_name="cascade_test",
            model_id=test_model.id,
            deployment_url="http://localhost:3001",
            status="pending",
            configuration={},
            replicas=1
        )
        test_session.add(deployment)
        test_session.commit()
        
        # Create prediction log
        log = PredictionLogDB(
            id="test_cascade_log",
            model_id=test_model.id,
            request_id="req_123",
            input_data={"test": "data"},
            output_data={"result": "test"},
            latency_ms=100.0,
            api_endpoint="/predict",
            success=True
        )
        test_session.add(log)
        test_session.commit()
        
        # Verify relationships exist
        test_session.refresh(test_model)
        assert len(test_model.deployments) > 0
        assert deployment in test_model.deployments
        
        # Verify monitoring data references the model correctly
        assert log.model_id == test_model.id
        
        # Test that we can query related data
        model_logs = test_session.query(PredictionLogDB).filter(
            PredictionLogDB.model_id == test_model.id
        ).all()
        assert len(model_logs) == 1
        assert model_logs[0].id == "test_cascade_log"
    
    def test_model_performance_tracking(self, test_session, test_model):
        """Test performance tracking across multiple metrics"""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        
        metrics_data = [
            ("accuracy", 0.95),
            ("precision", 0.92),
            ("recall", 0.98),
            ("f1_score", 0.95)
        ]
        
        # Create performance metrics
        for i, (metric_name, value) in enumerate(metrics_data):
            metric = ModelPerformanceMetricsDB(
                id=f"test_metric_{i}",
                model_id=test_model.id,
                time_window_start=now - timedelta(hours=1),
                time_window_end=now,
                total_requests=100,
                successful_requests=95,
                failed_requests=5,
                requests_per_minute=10.0,
                avg_latency_ms=150.0,
                p50_latency_ms=140.0,
                p95_latency_ms=200.0,
                p99_latency_ms=250.0,
                max_latency_ms=300.0,
                success_rate=value,  # Use the metric value as success rate for testing
                error_rate=0.05
            )
            test_session.add(metric)
        
        test_session.commit()
        
        # Query all metrics for the model
        model_metrics = test_session.query(ModelPerformanceMetricsDB).filter(
            ModelPerformanceMetricsDB.model_id == test_model.id
        ).all()
        
        assert len(model_metrics) == 4
        success_rates = [m.success_rate for m in model_metrics]
        assert 0.95 in success_rates
        assert 0.92 in success_rates
        assert 0.98 in success_rates
        assert 0.95 in success_rates 