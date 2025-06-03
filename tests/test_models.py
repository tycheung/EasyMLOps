"""
Unit tests for database models
Tests model validation, relationships, and serialization
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.model import Model, Deployment
from app.models.monitoring import (
    PredictionLog, ModelPerformanceMetric, SystemHealthMetric,
    Alert, AuditLog
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
            "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            "output_schema": {"type": "object", "properties": {"y": {"type": "number"}}},
            "performance_metrics": {"accuracy": 0.95}
        }
        
        model = Model(**model_data)
        test_session.add(model)
        test_session.commit()
        
        assert model.id is not None
        assert model.name == "test_model"
        assert model.model_type == "classification"
        assert model.framework == "sklearn"
        assert model.is_active is True
        assert model.created_at is not None
        assert model.updated_at is not None
    
    def test_model_validation_required_fields(self, test_session):
        """Test model validation for required fields"""
        # Missing required fields should raise error
        with pytest.raises(Exception):  # Could be IntegrityError or ValidationError
            model = Model()
            test_session.add(model)
            test_session.commit()
    
    def test_model_unique_name_constraint(self, test_session):
        """Test unique name constraint"""
        model1 = Model(
            name="duplicate_name",
            description="First model",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            input_schema={},
            output_schema={}
        )
        test_session.add(model1)
        test_session.commit()
        
        # Try to create another model with same name
        model2 = Model(
            name="duplicate_name",
            description="Second model",
            model_type="regression",
            framework="tensorflow",
            version="2.0.0",
            input_schema={},
            output_schema={}
        )
        
        test_session.add(model2)
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_model_json_fields(self, test_session):
        """Test JSON field storage and retrieval"""
        input_schema = {
            "type": "object",
            "properties": {
                "feature1": {"type": "number", "minimum": 0},
                "feature2": {"type": "string", "enum": ["A", "B", "C"]}
            },
            "required": ["feature1", "feature2"]
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "prediction": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.98,
            "f1_score": 0.95,
            "training_time": 120.5
        }
        
        model = Model(
            name="json_test_model",
            description="Testing JSON fields",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            input_schema=input_schema,
            output_schema=output_schema,
            performance_metrics=metrics
        )
        
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Verify JSON fields are stored and retrieved correctly
        assert model.input_schema == input_schema
        assert model.output_schema == output_schema
        assert model.performance_metrics == metrics
        assert model.performance_metrics["accuracy"] == 0.95
    
    def test_model_relationships(self, test_session):
        """Test model relationships with deployments"""
        model = Model(
            name="relationship_test",
            description="Testing relationships",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            input_schema={},
            output_schema={}
        )
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Create deployments for this model
        deployment1 = Deployment(
            name="deployment1",
            model_id=model.id,
            endpoint_url="http://localhost:3001",
            environment="test",
            resources={},
            scaling={}
        )
        
        deployment2 = Deployment(
            name="deployment2",
            model_id=model.id,
            endpoint_url="http://localhost:3002",
            environment="staging",
            resources={},
            scaling={}
        )
        
        test_session.add_all([deployment1, deployment2])
        test_session.commit()
        
        # Test relationship
        test_session.refresh(model)
        assert len(model.deployments) == 2
        assert deployment1 in model.deployments
        assert deployment2 in model.deployments
    
    def test_model_soft_delete(self, test_session):
        """Test model soft delete functionality"""
        model = Model(
            name="soft_delete_test",
            description="Testing soft delete",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            input_schema={},
            output_schema={}
        )
        test_session.add(model)
        test_session.commit()
        
        # Initially active
        assert model.is_active is True
        
        # Soft delete
        model.is_active = False
        test_session.commit()
        
        # Verify soft delete
        test_session.refresh(model)
        assert model.is_active is False
    
    def test_model_timestamps(self, test_session):
        """Test model timestamp behavior"""
        model = Model(
            name="timestamp_test",
            description="Testing timestamps",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
            input_schema={},
            output_schema={}
        )
        
        creation_time = datetime.now(timezone.utc)
        test_session.add(model)
        test_session.commit()
        test_session.refresh(model)
        
        # Check creation timestamp
        assert model.created_at is not None
        assert model.updated_at is not None
        assert model.created_at >= creation_time
        
        # Update model and check updated timestamp
        original_updated = model.updated_at
        model.description = "Updated description"
        test_session.commit()
        test_session.refresh(model)
        
        assert model.updated_at > original_updated


class TestDeploymentClass:
    """Test Deployment database model"""
    
    def test_deployment_creation(self, test_session, test_model):
        """Test basic deployment creation"""
        deployment_data = {
            "name": "test_deployment",
            "description": "A test deployment",
            "model_id": test_model.id,
            "endpoint_url": "http://localhost:3001",
            "environment": "test",
            "resources": {"cpu": "100m", "memory": "256Mi"},
            "scaling": {"min_replicas": 1, "max_replicas": 3}
        }
        
        deployment = Deployment(**deployment_data)
        test_session.add(deployment)
        test_session.commit()
        
        assert deployment.id is not None
        assert deployment.name == "test_deployment"
        assert deployment.model_id == test_model.id
        assert deployment.endpoint_url == "http://localhost:3001"
        assert deployment.status == "pending"
        assert deployment.is_active is True
        assert deployment.created_at is not None
    
    def test_deployment_model_relationship(self, test_session, test_model):
        """Test deployment-model relationship"""
        deployment = Deployment(
            name="relationship_test",
            model_id=test_model.id,
            endpoint_url="http://localhost:3001",
            environment="test",
            resources={},
            scaling={}
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
            deployment = Deployment(
                name=f"status_test_{status}",
                model_id=test_model.id,
                endpoint_url=f"http://localhost:300{valid_statuses.index(status)}",
                environment="test",
                status=status,
                resources={},
                scaling={}
            )
            test_session.add(deployment)
            test_session.commit()
            test_session.refresh(deployment)
            
            assert deployment.status == status
    
    def test_deployment_environment_values(self, test_session, test_model):
        """Test deployment environment values"""
        environments = ["development", "test", "staging", "production"]
        
        for env in environments:
            deployment = Deployment(
                name=f"env_test_{env}",
                model_id=test_model.id,
                endpoint_url=f"http://localhost:300{environments.index(env)}",
                environment=env,
                resources={},
                scaling={}
            )
            test_session.add(deployment)
            test_session.commit()
            test_session.refresh(deployment)
            
            assert deployment.environment == env
    
    def test_deployment_json_configuration(self, test_session, test_model):
        """Test JSON configuration storage"""
        resources = {
            "cpu": "500m",
            "memory": "1Gi",
            "gpu": {"type": "nvidia-tesla-k80", "count": 1}
        }
        
        scaling = {
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70,
            "scale_down_delay": "300s"
        }
        
        deployment = Deployment(
            name="json_config_test",
            model_id=test_model.id,
            endpoint_url="http://localhost:3001",
            environment="test",
            resources=resources,
            scaling=scaling
        )
        
        test_session.add(deployment)
        test_session.commit()
        test_session.refresh(deployment)
        
        assert deployment.resources == resources
        assert deployment.scaling == scaling
        assert deployment.resources["cpu"] == "500m"
        assert deployment.scaling["min_replicas"] == 2


class TestPredictionLog:
    """Test PredictionLog monitoring model"""
    
    def test_prediction_log_creation(self, test_session, test_model):
        """Test prediction log creation"""
        log_data = {
            "model_id": test_model.id,
            "input_data": {"feature1": 0.5, "feature2": "test"},
            "prediction": {"class": "A", "probability": 0.85},
            "response_time_ms": 150.5,
            "status": "success"
        }
        
        log = PredictionLog(**log_data)
        test_session.add(log)
        test_session.commit()
        
        assert log.id is not None
        assert log.model_id == test_model.id
        assert log.input_data == {"feature1": 0.5, "feature2": "test"}
        assert log.prediction == {"class": "A", "probability": 0.85}
        assert log.response_time_ms == 150.5
        assert log.status == "success"
        assert log.timestamp is not None
    
    def test_prediction_log_status_values(self, test_session, test_model):
        """Test prediction log status values"""
        statuses = ["success", "error", "timeout"]
        
        for status in statuses:
            log = PredictionLog(
                model_id=test_model.id,
                input_data={"test": "data"},
                prediction={"result": "test"},
                response_time_ms=100.0,
                status=status
            )
            test_session.add(log)
            test_session.commit()
            test_session.refresh(log)
            
            assert log.status == status
    
    def test_prediction_log_model_relationship(self, test_session, test_model):
        """Test prediction log model relationship"""
        log = PredictionLog(
            model_id=test_model.id,
            input_data={"test": "data"},
            prediction={"result": "test"},
            response_time_ms=100.0,
            status="success"
        )
        test_session.add(log)
        test_session.commit()
        test_session.refresh(log)
        
        assert log.model.id == test_model.id
        assert log.model.name == test_model.name


class TestModelPerformanceMetric:
    """Test ModelPerformanceMetric monitoring model"""
    
    def test_performance_metric_creation(self, test_session, test_model):
        """Test performance metric creation"""
        metric = ModelPerformanceMetric(
            model_id=test_model.id,
            metric_name="accuracy",
            metric_value=0.95,
            evaluation_data={"test_samples": 1000, "validation_set": "v1"}
        )
        
        test_session.add(metric)
        test_session.commit()
        
        assert metric.id is not None
        assert metric.model_id == test_model.id
        assert metric.metric_name == "accuracy"
        assert metric.metric_value == 0.95
        assert metric.evaluation_data == {"test_samples": 1000, "validation_set": "v1"}
        assert metric.timestamp is not None
    
    def test_performance_metric_types(self, test_session, test_model):
        """Test different performance metric types"""
        metrics = [
            ("accuracy", 0.95),
            ("precision", 0.92),
            ("recall", 0.98),
            ("f1_score", 0.95),
            ("auc_roc", 0.89),
            ("mean_squared_error", 0.05)
        ]
        
        for metric_name, value in metrics:
            metric = ModelPerformanceMetric(
                model_id=test_model.id,
                metric_name=metric_name,
                metric_value=value,
                evaluation_data={"test": "data"}
            )
            test_session.add(metric)
            test_session.commit()
            test_session.refresh(metric)
            
            assert metric.metric_name == metric_name
            assert metric.metric_value == value


class TestSystemHealthMetric:
    """Test SystemHealthMetric monitoring model"""
    
    def test_system_health_metric_creation(self, test_session):
        """Test system health metric creation"""
        metric = SystemHealthMetric(
            component_name="api_server",
            metric_name="cpu_usage",
            metric_value=45.2,
            unit="percentage",
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        
        test_session.add(metric)
        test_session.commit()
        
        assert metric.id is not None
        assert metric.component_name == "api_server"
        assert metric.metric_name == "cpu_usage"
        assert metric.metric_value == 45.2
        assert metric.unit == "percentage"
        assert metric.threshold_warning == 70.0
        assert metric.threshold_critical == 90.0
        assert metric.timestamp is not None
    
    def test_system_health_metric_components(self, test_session):
        """Test different system components"""
        components = [
            ("api_server", "cpu_usage", 45.2),
            ("database", "connection_pool", 80.0),
            ("redis", "memory_usage", 30.5),
            ("nginx", "requests_per_second", 150.0)
        ]
        
        for component, metric_name, value in components:
            metric = SystemHealthMetric(
                component_name=component,
                metric_name=metric_name,
                metric_value=value,
                unit="percentage"
            )
            test_session.add(metric)
            test_session.commit()
            test_session.refresh(metric)
            
            assert metric.component_name == component
            assert metric.metric_name == metric_name
            assert metric.metric_value == value


class TestAlert:
    """Test Alert monitoring model"""
    
    def test_alert_creation(self, test_session):
        """Test alert creation"""
        alert = Alert(
            alert_type="performance",
            severity="warning",
            title="High CPU Usage",
            message="CPU usage exceeded 70% threshold",
            source_component="api_server",
            metadata={"cpu_usage": 75.5, "threshold": 70.0}
        )
        
        test_session.add(alert)
        test_session.commit()
        
        assert alert.id is not None
        assert alert.alert_type == "performance"
        assert alert.severity == "warning"
        assert alert.title == "High CPU Usage"
        assert alert.is_resolved is False
        assert alert.created_at is not None
    
    def test_alert_severity_levels(self, test_session):
        """Test alert severity levels"""
        severities = ["info", "warning", "error", "critical"]
        
        for severity in severities:
            alert = Alert(
                alert_type="test",
                severity=severity,
                title=f"Test {severity} alert",
                message=f"This is a {severity} level alert",
                source_component="test_component"
            )
            test_session.add(alert)
            test_session.commit()
            test_session.refresh(alert)
            
            assert alert.severity == severity
    
    def test_alert_resolution(self, test_session):
        """Test alert resolution"""
        alert = Alert(
            alert_type="test",
            severity="warning",
            title="Test Alert",
            message="Test message",
            source_component="test"
        )
        
        test_session.add(alert)
        test_session.commit()
        
        # Initially unresolved
        assert alert.is_resolved is False
        assert alert.resolved_at is None
        
        # Resolve alert
        alert.is_resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        test_session.commit()
        test_session.refresh(alert)
        
        assert alert.is_resolved is True
        assert alert.resolved_at is not None


class TestAuditLog:
    """Test AuditLog monitoring model"""
    
    def test_audit_log_creation(self, test_session):
        """Test audit log creation"""
        audit = AuditLog(
            action="model_upload",
            resource_type="model",
            resource_id="model_123",
            user_id="user_456",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Test Browser)",
            changes={"status": {"from": "pending", "to": "active"}}
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
        
        for action, resource_type in actions:
            audit = AuditLog(
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
        """Test cascade behavior when model is deleted"""
        # Create deployment
        deployment = Deployment(
            name="cascade_test",
            model_id=test_model.id,
            endpoint_url="http://localhost:3001",
            environment="test",
            resources={},
            scaling={}
        )
        test_session.add(deployment)
        test_session.commit()
        
        # Create prediction log
        log = PredictionLog(
            model_id=test_model.id,
            input_data={"test": "data"},
            prediction={"result": "test"},
            response_time_ms=100.0,
            status="success"
        )
        test_session.add(log)
        test_session.commit()
        
        # Verify relationships exist
        assert len(test_model.deployments) > 0
        
        # Delete model (in real app, this might be soft delete)
        test_session.delete(test_model)
        test_session.commit()
        
        # Check what happens to related records
        # (Behavior depends on cascade configuration)
    
    def test_model_performance_tracking(self, test_session, test_model):
        """Test performance tracking across multiple metrics"""
        metrics = [
            ("accuracy", 0.95),
            ("precision", 0.92),
            ("recall", 0.98),
            ("f1_score", 0.95)
        ]
        
        for metric_name, value in metrics:
            metric = ModelPerformanceMetric(
                model_id=test_model.id,
                metric_name=metric_name,
                metric_value=value,
                evaluation_data={"timestamp": datetime.now().isoformat()}
            )
            test_session.add(metric)
        
        test_session.commit()
        
        # Query all metrics for the model
        model_metrics = test_session.query(ModelPerformanceMetric).filter(
            ModelPerformanceMetric.model_id == test_model.id
        ).all()
        
        assert len(model_metrics) == 4
        metric_names = [m.metric_name for m in model_metrics]
        assert "accuracy" in metric_names
        assert "precision" in metric_names
        assert "recall" in metric_names
        assert "f1_score" in metric_names 