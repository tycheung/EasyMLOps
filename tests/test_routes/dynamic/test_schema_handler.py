"""
Tests for dynamic route schema handler
Tests schema information endpoint for deployed models
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.models.model import ModelDeployment
from app.routes.dynamic.schema_handler import get_prediction_schema
from app.schemas.model import InputSchema, OutputSchema, FieldSchema, DataType


class TestSchemaHandler:
    """Test schema handler endpoint"""
    
    @pytest.fixture
    def sample_deployment(self, test_model):
        """Sample deployment for testing"""
        return ModelDeployment(
            id="deploy_123",
            model_id=test_model.id,
            deployment_name="test_deployment",
            deployment_url="http://localhost:3000/model_service_123",
            status="active",
            configuration={},
            framework="sklearn",
            endpoints=["predict", "predict_proba"],
            created_at=None
        )
    
    @pytest.fixture
    def mock_input_schema(self):
        """Mock input schema"""
        return InputSchema(
            fields=[
                FieldSchema(
                    name="bedrooms",
                    data_type=DataType.INTEGER,
                    description="Number of bedrooms",
                    required=True
                ),
                FieldSchema(
                    name="bathrooms",
                    data_type=DataType.FLOAT,
                    description="Number of bathrooms",
                    required=True
                )
            ],
            batch_input=False
        )
    
    @pytest.fixture
    def mock_output_schema(self):
        """Mock output schema"""
        return OutputSchema(
            fields=[
                FieldSchema(
                    name="predicted_price",
                    data_type=DataType.FLOAT,
                    description="Predicted price",
                    required=True
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_success(
        self, isolated_test_session, sample_deployment, mock_input_schema, mock_output_schema
    ):
        """Test successfully getting prediction schema"""
        isolated_test_session.add(sample_deployment)
        await isolated_test_session.commit()
        
        example_data = {"bedrooms": 3, "bathrooms": 2.5}
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_get_schemas.return_value = (mock_input_schema, mock_output_schema)
            mock_get_example.return_value = example_data
            
            result = await get_prediction_schema(
                sample_deployment.id,
                session=isolated_test_session
            )
            
            assert result["deployment_id"] == sample_deployment.id
            assert result["model_id"] == sample_deployment.model_id
            assert result["framework"] == "sklearn"
            assert result["endpoints"] == ["predict", "predict_proba"]
            assert result["input_schema"] is not None
            assert result["output_schema"] is not None
            assert result["example_input"] == example_data
            assert result["validation_enabled"] is True
            assert "description" in result
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_deployment_not_found(self, isolated_isolated_test_session):
        """Test getting schema for non-existent deployment"""
        with pytest.raises(HTTPException) as exc_info:
            await get_prediction_schema(
                "nonexistent_deployment",
                session=isolated_isolated_test_session
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_no_input_schema(
        self, isolated_test_session, sample_deployment, mock_output_schema
    ):
        """Test getting schema when input schema is missing"""
        isolated_test_session.add(sample_deployment)
        await isolated_test_session.commit()
        
        example_data = {"bedrooms": 3, "bathrooms": 2.5}
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_get_schemas.return_value = (None, mock_output_schema)
            mock_get_example.return_value = example_data
            
            result = await get_prediction_schema(
                sample_deployment.id,
                session=isolated_test_session
            )
            
            assert result["input_schema"] is None
            assert result["output_schema"] is not None
            assert result["validation_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_no_output_schema(
        self, isolated_test_session, sample_deployment, mock_input_schema
    ):
        """Test getting schema when output schema is missing"""
        isolated_test_session.add(sample_deployment)
        await isolated_test_session.commit()
        
        example_data = {"bedrooms": 3, "bathrooms": 2.5}
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_get_schemas.return_value = (mock_input_schema, None)
            mock_get_example.return_value = example_data
            
            result = await get_prediction_schema(
                sample_deployment.id,
                session=isolated_test_session
            )
            
            assert result["input_schema"] is not None
            assert result["output_schema"] is None
            assert result["validation_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_no_schemas(
        self, isolated_test_session, sample_deployment
    ):
        """Test getting schema when both schemas are missing"""
        isolated_test_session.add(sample_deployment)
        await isolated_test_session.commit()
        
        example_data = {"bedrooms": 3, "bathrooms": 2.5}
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_get_schemas.return_value = (None, None)
            mock_get_example.return_value = example_data
            
            result = await get_prediction_schema(
                sample_deployment.id,
                session=isolated_test_session
            )
            
            assert result["input_schema"] is None
            assert result["output_schema"] is None
            assert result["validation_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_service_error(
        self, isolated_test_session, sample_deployment
    ):
        """Test error handling when schema service fails"""
        isolated_test_session.add(sample_deployment)
        await isolated_test_session.commit()
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas:
            mock_get_schemas.side_effect = Exception("Schema service error")
            
            with pytest.raises(HTTPException) as exc_info:
                await get_prediction_schema(
                    sample_deployment.id,
                    session=isolated_test_session
                )
            
            assert exc_info.value.status_code == 500
            assert "Failed to get prediction schema" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_get_prediction_schema_different_frameworks(
        self, isolated_test_session, test_model, mock_input_schema, mock_output_schema
    ):
        """Test schema handler with different frameworks"""
        frameworks = ["sklearn", "tensorflow", "pytorch", "xgboost"]
        
        for framework in frameworks:
            deployment = ModelDeployment(
                id=f"deploy_{framework}",
                model_id=test_model.id,
                deployment_name=f"deployment_{framework}",
                deployment_url=f"http://localhost:3000/{framework}",
                status="active",
                configuration={},
                framework=framework,
                endpoints=["predict"],
                created_at=None
            )
            isolated_test_session.add(deployment)
        
        await isolated_test_session.commit()
        
        example_data = {"bedrooms": 3, "bathrooms": 2.5}
        
        with patch('app.services.schema_service.schema_service.get_model_schemas') as mock_get_schemas, \
             patch('app.services.schema_service.schema_service.get_model_example_data') as mock_get_example:
            
            mock_get_schemas.return_value = (mock_input_schema, mock_output_schema)
            mock_get_example.return_value = example_data
            
            for framework in frameworks:
                result = await get_prediction_schema(
                    f"deploy_{framework}",
                    session=isolated_test_session
                )
                
                assert result["framework"] == framework
                assert result["deployment_id"] == f"deploy_{framework}"

