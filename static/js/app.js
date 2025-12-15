// EasyMLOps Frontend Application
// Handles all user interactions, API calls, and dynamic content

// API Configuration
const API_BASE = '/api/v1';

// Application State
let currentModels = [];
let currentDeployments = [];
let currentTab = 'dashboard';
let inputFieldCount = 0;
let outputFieldCount = 0;

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    console.log('EasyMLOps Frontend Initialized');
    setupFileUpload();
    setupForms();
    loadDashboardData();
    loadCustomTemplates(); // Load custom templates from localStorage
    
    // Auto-refresh dashboard every 30 seconds
    setInterval(loadDashboardData, 30000);
});

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
        currentTab = tabName;
        
        // Load tab-specific data
        switch(tabName) {
            case 'dashboard':
                loadDashboardData();
                break;
            case 'models':
                loadModels();
                break;
            case 'deployments':
                loadDeployments();
                break;
            case 'test':
                loadTestModels();
                break;
            case 'monitoring':
                loadMonitoringData();
                populateMonitoringModelSelectors();
                break;
            case 'schemas':
                // Schema tab initialized - forms are set up in DOMContentLoaded
                break;
            case 'ab-testing':
                loadABTests();
                break;
            case 'canary':
                loadCanaryDeployments();
                break;
            case 'governance':
                // Governance tab initialized
                break;
            case 'analytics':
                // Analytics tab initialized - forms are set up in DOMContentLoaded
                populateMonitoringModelSelectors(); // For time-series model selector
                break;
            case 'lifecycle':
                populateMonitoringModelSelectors();
                break;
            case 'integrations':
                // Integrations tab initialized
                break;
            case 'audit':
                loadAuditLogs();
                break;
        }
    }
}

// Notification System
function showNotification(message, type = 'info') {
    const notifications = document.getElementById('notifications');
    const notification = document.createElement('div');
    
    const bgColor = type === 'success' ? 'bg-green-500' : 
                   type === 'error' ? 'bg-red-500' : 
                   type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500';
    
    notification.innerHTML = `
        <div class="${bgColor} text-white px-6 py-4 rounded-lg shadow-lg mb-4">
            <div class="flex items-center justify-between">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="text-white hover:text-gray-200">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
    
    notifications.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Loading Overlay
function showLoading() {
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

// API Helper Functions
async function apiCall(endpoint, options = {}) {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.error?.message || 'API call failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        showNotification(error.message, 'error');
        throw error;
    } finally {
        hideLoading();
    }
}

// Dashboard Functions
async function loadDashboardData() {
    try {
        // Load models
        const models = await apiCall('/models/');
        currentModels = models;
        document.getElementById('total-models').textContent = models.length;
        
        // Load deployments
        const deployments = await apiCall('/deployments/');
        currentDeployments = deployments;
        const activeDeployments = deployments.filter(d => d.status === 'active').length;
        document.getElementById('active-deployments').textContent = activeDeployments;
        
        // Load real monitoring metrics
        try {
            const dashboardMetrics = await apiCall('/monitoring/dashboard');
            document.getElementById('predictions-today').textContent = dashboardMetrics.total_predictions || 0;
            document.getElementById('avg-response-time').textContent = `${Math.round(dashboardMetrics.avg_latency_ms || 0)}ms`;
        } catch (error) {
            console.warn('Could not load monitoring metrics, using defaults:', error);
            // Fallback to defaults if monitoring service unavailable
            document.getElementById('predictions-today').textContent = '0';
            document.getElementById('avg-response-time').textContent = '0ms';
        }
        
        // Update recent activity
        updateRecentActivity();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateRecentActivity() {
    const activityContainer = document.getElementById('recent-activity');
    const activities = [];
    
    // Add model activities
    currentModels.slice(0, 3).forEach(model => {
        activities.push({
            type: 'model',
            message: `Model "${model.name}" uploaded`,
            time: new Date(model.created_at).toLocaleString(),
            icon: 'fa-brain',
            color: 'text-blue-600'
        });
    });
    
    // Add deployment activities
    currentDeployments.slice(0, 2).forEach(deployment => {
        activities.push({
            type: 'deployment',
            message: `Model deployed: ${deployment.service_name}`,
            time: new Date(deployment.created_at).toLocaleString(),
            icon: 'fa-rocket',
            color: 'text-green-600'
        });
    });
    
    if (activities.length === 0) {
        activityContainer.innerHTML = `
            <div class="text-gray-500 text-center py-8">
                <i class="fas fa-history text-3xl mb-4"></i>
                <p>No recent activity</p>
            </div>
        `;
    } else {
        activityContainer.innerHTML = activities.map(activity => `
            <div class="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <i class="fas ${activity.icon} ${activity.color}"></i>
                <div class="flex-1">
                    <p class="text-sm text-gray-800">${activity.message}</p>
                    <p class="text-xs text-gray-500">${activity.time}</p>
                </div>
            </div>
        `).join('');
    }
}

// File Upload Setup
function setupFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('model-file');
    
    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            displaySelectedFile(files[0]);
        }
    });
    
    // File selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            displaySelectedFile(e.target.files[0]);
        }
    });
}

function displaySelectedFile(file) {
    const uploadContent = document.getElementById('upload-content');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    
    uploadContent.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
}

function clearFile() {
    const uploadContent = document.getElementById('upload-content');
    const fileInfo = document.getElementById('file-info');
    const fileInput = document.getElementById('model-file');
    
    fileInfo.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    fileInput.value = '';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Schema Configuration Functions
function toggleSchemaConfig() {
    const schemaConfig = document.getElementById('schema-config');
    const toggleButton = document.getElementById('schema-toggle');
    
    if (schemaConfig.classList.contains('hidden')) {
        schemaConfig.classList.remove('hidden');
        toggleButton.innerHTML = '<i class="fas fa-minus mr-2"></i>Hide Schema';
    } else {
        schemaConfig.classList.add('hidden');
        toggleButton.innerHTML = '<i class="fas fa-plus mr-2"></i>Configure Schema';
    }
}

function addInputField() {
    const container = document.getElementById('input-fields');
    const fieldId = `input-field-${inputFieldCount++}`;
    
    const fieldHtml = createFieldConfigHTML(fieldId, 'input');
    container.insertAdjacentHTML('beforeend', fieldHtml);
}

function addOutputField() {
    const container = document.getElementById('output-fields');
    const fieldId = `output-field-${outputFieldCount++}`;
    
    const fieldHtml = createFieldConfigHTML(fieldId, 'output');
    container.insertAdjacentHTML('beforeend', fieldHtml);
}

function createFieldConfigHTML(fieldId, type) {
    const isInput = type === 'input';
    return `
        <div class="border rounded-lg p-4 bg-gray-50" id="${fieldId}">
            <div class="flex justify-between items-start mb-3">
                <h6 class="font-medium text-gray-900">${isInput ? 'Input' : 'Output'} Field</h6>
                <button type="button" onclick="removeField('${fieldId}')" class="text-red-600 hover:text-red-800">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                    <label class="block text-xs font-medium text-gray-700 mb-1">Field Name</label>
                    <input type="text" class="field-name w-full text-sm border-gray-300 rounded-md" placeholder="field_name" required>
                </div>
                <div>
                    <label class="block text-xs font-medium text-gray-700 mb-1">Data Type</label>
                    <select class="field-type w-full text-sm border-gray-300 rounded-md" required>
                        <option value="string">String</option>
                        <option value="integer">Integer</option>
                        <option value="float">Float</option>
                        <option value="boolean">Boolean</option>
                        <option value="array">Array</option>
                        <option value="object">Object</option>
                        <option value="date">Date</option>
                        <option value="datetime">DateTime</option>
                    </select>
                </div>
            </div>
            
            <div class="mt-3">
                <label class="block text-xs font-medium text-gray-700 mb-1">Description</label>
                <input type="text" class="field-description w-full text-sm border-gray-300 rounded-md" placeholder="Field description">
            </div>
            
            ${isInput ? `
            <div class="mt-3">
                <label class="flex items-center">
                    <input type="checkbox" class="field-required rounded border-gray-300" checked>
                    <span class="ml-2 text-xs text-gray-700">Required</span>
                </label>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                <div>
                    <label class="block text-xs font-medium text-gray-700 mb-1">Min Value</label>
                    <input type="number" step="any" class="field-min w-full text-sm border-gray-300 rounded-md" placeholder="Optional">
                </div>
                <div>
                    <label class="block text-xs font-medium text-gray-700 mb-1">Max Value</label>
                    <input type="number" step="any" class="field-max w-full text-sm border-gray-300 rounded-md" placeholder="Optional">
                </div>
            </div>
            ` : ''}
        </div>
    `;
}

function removeField(fieldId) {
    const field = document.getElementById(fieldId);
    if (field) {
        field.remove();
    }
}

// Schema Template Loading
async function loadTemplate(templateName) {
    try {
        const templates = getBuiltInTemplates();
        const template = templates[templateName];
        
        if (!template) {
            showNotification('Template not found', 'error');
            return;
        }
        
        // Clear existing fields
        document.getElementById('input-fields').innerHTML = '';
        document.getElementById('output-fields').innerHTML = '';
        inputFieldCount = 0;
        outputFieldCount = 0;
        
        // Load input fields
        template.input_schema.fields.forEach(field => {
            addInputField();
            const fieldElement = document.getElementById(`input-field-${inputFieldCount - 1}`);
            populateFieldFromTemplate(fieldElement, field, true);
        });
        
        // Load output fields
        template.output_schema.fields.forEach(field => {
            addOutputField();
            const fieldElement = document.getElementById(`output-field-${outputFieldCount - 1}`);
            populateFieldFromTemplate(fieldElement, field, false);
        });
        
        // Show schema config if hidden
        const schemaConfig = document.getElementById('schema-config');
        if (schemaConfig.classList.contains('hidden')) {
            toggleSchemaConfig();
        }
        
        showNotification(`Loaded ${template.name} template`, 'success');
        
    } catch (error) {
        console.error('Error loading template:', error);
        showNotification('Error loading template: ' + error.message, 'error');
    }
}

// Built-in Schema Templates
function getBuiltInTemplates() {
    return {
        'house_price_prediction': {
            name: 'House Price Prediction',
            description: 'Real estate price prediction model template',
            input_schema: {
                fields: [
                    {
                        name: 'square_feet',
                        data_type: 'number',
                        description: 'Total square footage of the house',
                        required: true,
                        min_value: 500,
                        max_value: 10000
                    },
                    {
                        name: 'bedrooms',
                        data_type: 'integer',
                        description: 'Number of bedrooms',
                        required: true,
                        min_value: 1,
                        max_value: 10
                    },
                    {
                        name: 'bathrooms',
                        data_type: 'number',
                        description: 'Number of bathrooms',
                        required: true,
                        min_value: 1,
                        max_value: 10
                    },
                    {
                        name: 'age',
                        data_type: 'integer',
                        description: 'Age of the house in years',
                        required: true,
                        min_value: 0,
                        max_value: 100
                    },
                    {
                        name: 'location_score',
                        data_type: 'number',
                        description: 'Location desirability score (1-10)',
                        required: false,
                        min_value: 1,
                        max_value: 10
                    }
                ]
            },
            output_schema: {
                fields: [
                    {
                        name: 'predicted_price',
                        data_type: 'number',
                        description: 'Predicted house price in USD'
                    },
                    {
                        name: 'confidence_score',
                        data_type: 'number',
                        description: 'Model confidence (0-1)'
                    },
                    {
                        name: 'price_range',
                        data_type: 'object',
                        description: 'Price range estimate with min/max'
                    }
                ]
            }
        },
        'classification': {
            name: 'Binary Classification',
            description: 'General binary classification model template',
            input_schema: {
                fields: [
                    {
                        name: 'feature_1',
                        data_type: 'number',
                        description: 'First numerical feature',
                        required: true
                    },
                    {
                        name: 'feature_2',
                        data_type: 'number',
                        description: 'Second numerical feature',
                        required: true
                    },
                    {
                        name: 'feature_3',
                        data_type: 'number',
                        description: 'Third numerical feature',
                        required: false
                    },
                    {
                        name: 'category',
                        data_type: 'string',
                        description: 'Categorical feature',
                        required: false
                    }
                ]
            },
            output_schema: {
                fields: [
                    {
                        name: 'prediction',
                        data_type: 'string',
                        description: 'Predicted class (0 or 1)'
                    },
                    {
                        name: 'probability',
                        data_type: 'number',
                        description: 'Probability of positive class'
                    },
                    {
                        name: 'probabilities',
                        data_type: 'object',
                        description: 'Full probability distribution'
                    }
                ]
            }
        },
        'text_analysis': {
            name: 'Text Analysis',
            description: 'Text sentiment and classification template',
            input_schema: {
                fields: [
                    {
                        name: 'text',
                        data_type: 'string',
                        description: 'Input text to analyze',
                        required: true
                    },
                    {
                        name: 'language',
                        data_type: 'string',
                        description: 'Text language (optional)',
                        required: false
                    },
                    {
                        name: 'max_length',
                        data_type: 'integer',
                        description: 'Maximum text length to process',
                        required: false,
                        min_value: 10,
                        max_value: 5000
                    }
                ]
            },
            output_schema: {
                fields: [
                    {
                        name: 'sentiment',
                        data_type: 'string',
                        description: 'Predicted sentiment (positive/negative/neutral)'
                    },
                    {
                        name: 'confidence',
                        data_type: 'number',
                        description: 'Sentiment prediction confidence'
                    },
                    {
                        name: 'emotions',
                        data_type: 'object',
                        description: 'Detected emotions with scores'
                    },
                    {
                        name: 'keywords',
                        data_type: 'array',
                        description: 'Extracted keywords from text'
                    }
                ]
            }
        }
    };
}

function populateFieldFromTemplate(fieldElement, fieldData, isInput) {
    fieldElement.querySelector('.field-name').value = fieldData.name;
    fieldElement.querySelector('.field-type').value = fieldData.data_type;
    fieldElement.querySelector('.field-description').value = fieldData.description || '';
    
    if (isInput) {
        const requiredCheckbox = fieldElement.querySelector('.field-required');
        if (requiredCheckbox) {
            requiredCheckbox.checked = fieldData.required;
        }
        
        const minField = fieldElement.querySelector('.field-min');
        const maxField = fieldElement.querySelector('.field-max');
        
        if (minField && fieldData.min_value !== undefined) {
            minField.value = fieldData.min_value;
        }
        if (maxField && fieldData.max_value !== undefined) {
            maxField.value = fieldData.max_value;
        }
    }
}

// Template Management Functions
function openSaveTemplateModal() {
    const inputFields = collectSchemaFields('input-fields');
    const outputFields = collectSchemaFields('output-fields');
    
    if (inputFields.length === 0 && outputFields.length === 0) {
        showNotification('Please add some schema fields before saving as template', 'warning');
        return;
    }
    
    // Update template preview
    updateTemplatePreview(inputFields, outputFields);
    
    // Show modal
    document.getElementById('save-template-modal').classList.remove('hidden');
}

function closeSaveTemplateModal() {
    document.getElementById('save-template-modal').classList.add('hidden');
    document.getElementById('save-template-form').reset();
    document.getElementById('template-preview').innerHTML = '';
}

function updateTemplatePreview(inputFields, outputFields) {
    const preview = document.getElementById('template-preview');
    preview.innerHTML = `
        <div class="border rounded p-3 bg-gray-50">
            <h4 class="font-medium text-sm mb-2">Template Preview:</h4>
            <div class="text-xs space-y-1">
                <div><span class="font-medium">Input fields:</span> ${inputFields.length} field(s)</div>
                <div><span class="font-medium">Output fields:</span> ${outputFields.length} field(s)</div>
                ${inputFields.length > 0 ? `<div class="text-blue-600">Input: ${inputFields.map(f => f.name).join(', ')}</div>` : ''}
                ${outputFields.length > 0 ? `<div class="text-green-600">Output: ${outputFields.map(f => f.name).join(', ')}</div>` : ''}
            </div>
        </div>
    `;
}

function saveTemplate(event) {
    event.preventDefault();
    
    const templateName = document.getElementById('template-name').value.trim();
    const templateDescription = document.getElementById('template-description').value.trim();
    
    if (!templateName) {
        showNotification('Please enter a template name', 'error');
        return;
    }
    
    const inputFields = collectSchemaFields('input-fields');
    const outputFields = collectSchemaFields('output-fields');
    
    if (inputFields.length === 0 && outputFields.length === 0) {
        showNotification('Cannot save empty template', 'error');
        return;
    }
    
    const template = {
        id: generateTemplateId(templateName),
        name: templateName,
        description: templateDescription,
        created_at: new Date().toISOString(),
        input_schema: { fields: inputFields },
        output_schema: { fields: outputFields }
    };
    
    try {
        // Save to localStorage (in future, this could be saved to backend)
        const customTemplates = getCustomTemplates();
        customTemplates[template.id] = template;
        localStorage.setItem('easymlops_custom_templates', JSON.stringify(customTemplates));
        
        // Update the UI
        loadCustomTemplates();
        
        // Close modal and show success
        closeSaveTemplateModal();
        showNotification(`Template "${templateName}" saved successfully!`, 'success');
        
    } catch (error) {
        console.error('Error saving template:', error);
        showNotification('Error saving template: ' + error.message, 'error');
    }
}

function generateTemplateId(name) {
    return 'custom_' + name.toLowerCase().replace(/[^a-z0-9]/g, '_');
}

function getCustomTemplates() {
    try {
        const stored = localStorage.getItem('easymlops_custom_templates');
        return stored ? JSON.parse(stored) : {};
    } catch (error) {
        console.error('Error loading custom templates:', error);
        return {};
    }
}

function loadCustomTemplates() {
    const customTemplates = getCustomTemplates();
    const container = document.getElementById('custom-templates');
    const section = document.getElementById('custom-templates-section');
    
    container.innerHTML = '';
    
    if (Object.keys(customTemplates).length === 0) {
        section.classList.add('hidden');
        return;
    }
    
    section.classList.remove('hidden');
    
    Object.values(customTemplates).forEach(template => {
        const button = document.createElement('button');
        button.type = 'button';
        button.onclick = () => loadCustomTemplate(template.id);
        button.className = 'bg-orange-100 hover:bg-orange-200 text-orange-800 px-3 py-1 rounded text-sm relative group';
        button.innerHTML = `
            <i class="fas fa-user mr-1"></i>${template.name}
            <button onclick="deleteCustomTemplate('${template.id}', event)" 
                    class="ml-2 text-orange-600 hover:text-red-600 opacity-0 group-hover:opacity-100 transition-opacity">
                <i class="fas fa-times text-xs"></i>
            </button>
        `;
        container.appendChild(button);
    });
}

function loadCustomTemplate(templateId) {
    try {
        const customTemplates = getCustomTemplates();
        const template = customTemplates[templateId];
        
        if (!template) {
            showNotification('Template not found', 'error');
            return;
        }
        
        // Clear existing fields
        document.getElementById('input-fields').innerHTML = '';
        document.getElementById('output-fields').innerHTML = '';
        inputFieldCount = 0;
        outputFieldCount = 0;
        
        // Load input fields
        template.input_schema.fields.forEach(field => {
            addInputField();
            const fieldElement = document.getElementById(`input-field-${inputFieldCount - 1}`);
            populateFieldFromTemplate(fieldElement, field, true);
        });
        
        // Load output fields
        template.output_schema.fields.forEach(field => {
            addOutputField();
            const fieldElement = document.getElementById(`output-field-${outputFieldCount - 1}`);
            populateFieldFromTemplate(fieldElement, field, false);
        });
        
        // Show schema config if hidden
        const schemaConfig = document.getElementById('schema-config');
        if (schemaConfig.classList.contains('hidden')) {
            toggleSchemaConfig();
        }
        
        showNotification(`Loaded custom template "${template.name}"`, 'success');
        
    } catch (error) {
        console.error('Error loading custom template:', error);
        showNotification('Error loading template: ' + error.message, 'error');
    }
}

function deleteCustomTemplate(templateId, event) {
    event.stopPropagation(); // Prevent template loading when delete button is clicked
    
    if (!confirm('Are you sure you want to delete this template?')) {
        return;
    }
    
    try {
        const customTemplates = getCustomTemplates();
        const templateName = customTemplates[templateId]?.name || 'Unknown';
        
        delete customTemplates[templateId];
        localStorage.setItem('easymlops_custom_templates', JSON.stringify(customTemplates));
        
        loadCustomTemplates();
        showNotification(`Template "${templateName}" deleted`, 'success');
        
    } catch (error) {
        console.error('Error deleting template:', error);
        showNotification('Error deleting template: ' + error.message, 'error');
    }
}

// Form Setup and Submission
function setupForms() {
    // Upload form
    const uploadForm = document.getElementById('upload-form');
    uploadForm.addEventListener('submit', handleModelUpload);
    
    // Test form
    const testForm = document.getElementById('test-form');
    testForm.addEventListener('submit', handleModelTest);
}

async function handleModelUpload(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('model-file');
    
    if (!fileInput.files[0]) {
        showNotification('Please select a model file', 'error');
        return;
    }
    
    // Collect form data
    const modelData = {
        name: document.getElementById('model-name').value,
        framework: document.getElementById('model-framework').value,
        model_type: document.getElementById('model-type').value,
        version: document.getElementById('model-version').value,
        description: document.getElementById('model-description').value
    };
    
    try {
        // Upload model file
        formData.append('file', fileInput.files[0]);
        Object.keys(modelData).forEach(key => {
            formData.append(key, modelData[key]);
        });
        
        const uploadResponse = await fetch(`${API_BASE}/models/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            const error = await uploadResponse.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const modelResult = await uploadResponse.json();
        const modelId = modelResult.model_id;
        
        // Upload schema if configured
        const schemaConfig = document.getElementById('schema-config');
        if (!schemaConfig.classList.contains('hidden')) {
            await uploadModelSchema(modelId);
        }
        
        showNotification('Model uploaded successfully!', 'success');
        
        // Reset form and switch to models tab
        uploadForm.reset();
        clearFile();
        showTab('models');
        
    } catch (error) {
        console.error('Upload error:', error);
        showNotification(error.message, 'error');
    }
}

async function uploadModelSchema(modelId) {
    const inputFields = collectSchemaFields('input-fields');
    const outputFields = collectSchemaFields('output-fields');
    
    if (inputFields.length === 0 && outputFields.length === 0) {
        return; // No schema to upload
    }
    
    const schemaData = {
        input_schema: { fields: inputFields },
        output_schema: { fields: outputFields }
    };
    
    try {
        await apiCall(`/models/${modelId}/schemas`, {
            method: 'POST',
            body: JSON.stringify(schemaData)
        });
        
        showNotification('Schema uploaded successfully!', 'success');
    } catch (error) {
        console.error('Schema upload error:', error);
        showNotification('Schema upload failed: ' + error.message, 'warning');
    }
}

function collectSchemaFields(containerId) {
    const container = document.getElementById(containerId);
    const fields = [];
    
    container.querySelectorAll('.border.rounded-lg').forEach(fieldElement => {
        const name = fieldElement.querySelector('.field-name').value;
        const dataType = fieldElement.querySelector('.field-type').value;
        const description = fieldElement.querySelector('.field-description').value;
        
        if (name && dataType) {
            const field = {
                name: name,
                data_type: dataType,
                description: description || null
            };
            
            // Add input-specific fields
            if (containerId === 'input-fields') {
                const requiredCheckbox = fieldElement.querySelector('.field-required');
                const minField = fieldElement.querySelector('.field-min');
                const maxField = fieldElement.querySelector('.field-max');
                
                field.required = requiredCheckbox ? requiredCheckbox.checked : true;
                
                if (minField && minField.value) {
                    field.min_value = parseFloat(minField.value);
                }
                if (maxField && maxField.value) {
                    field.max_value = parseFloat(maxField.value);
                }
            }
            
            fields.push(field);
        }
    });
    
    return fields;
}

// Models Management
async function loadModels() {
    try {
        const models = await apiCall('/models/');
        currentModels = models;
        renderModelsList(models);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function renderModelsList(models) {
    const container = document.getElementById('models-list');
    
    if (models.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-brain text-4xl mb-4"></i>
                <p>No models uploaded yet</p>
                <button onclick="showTab('upload')" class="mt-4 text-blue-600 hover:text-blue-800">
                    Upload your first model
                </button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = models.map(model => `
        <div class="border-b border-gray-200 py-4 last:border-b-0">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <h4 class="text-lg font-medium text-gray-900">${model.name}</h4>
                    <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                        <span><i class="fas fa-cogs mr-1"></i>${model.framework}</span>
                        <span><i class="fas fa-tag mr-1"></i>${model.model_type}</span>
                        <span><i class="fas fa-calendar mr-1"></i>${new Date(model.created_at).toLocaleDateString()}</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getStatusBadgeClass(model.status)}">${model.status}</span>
                    </div>
                    ${model.description ? `<p class="mt-2 text-gray-600">${model.description}</p>` : ''}
                </div>
                <div class="flex items-center space-x-2">
                    <button onclick="viewModelDetails('${model.id}')" class="text-blue-600 hover:text-blue-800" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="openUpdateModelModal('${model.id}')" class="text-yellow-600 hover:text-yellow-800" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button onclick="validateModel('${model.id}')" class="text-purple-600 hover:text-purple-800" title="Validate">
                        <i class="fas fa-check-circle"></i>
                    </button>
                    <button onclick="getModelMetrics('${model.id}')" class="text-indigo-600 hover:text-indigo-800" title="Metrics">
                        <i class="fas fa-chart-bar"></i>
                    </button>
                    <button onclick="deployModel('${model.id}')" class="text-green-600 hover:text-green-800" title="Deploy">
                        <i class="fas fa-rocket"></i>
                    </button>
                    <button onclick="deleteModel('${model.id}')" class="text-red-600 hover:text-red-800" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

function getStatusBadgeClass(status) {
    switch(status) {
        case 'uploaded': return 'bg-blue-100 text-blue-800';
        case 'validated': return 'bg-green-100 text-green-800';
        case 'error': return 'bg-red-100 text-red-800';
        default: return 'bg-gray-100 text-gray-800';
    }
}

async function deployModel(modelId) {
    try {
        const deployment = await apiCall(`/deployments/`, {
            method: 'POST',
            body: JSON.stringify({
                model_id: modelId,
                environment: 'production'
            })
        });
        
        showNotification('Model deployment started!', 'success');
        showTab('deployments');
        
    } catch (error) {
        console.error('Deployment error:', error);
    }
}

async function deleteModel(modelId) {
    if (!confirm('Are you sure you want to delete this model?')) {
        return;
    }
    
    try {
        await apiCall(`/models/${modelId}`, {
            method: 'DELETE'
        });
        
        showNotification('Model deleted successfully', 'success');
        loadModels();
        
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('Failed to delete model: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Model Update Functions
async function openUpdateModelModal(modelId) {
    try {
        const model = await apiCall(`/models/${modelId}`);
        const modal = document.getElementById('update-model-modal');
        if (modal) {
            document.getElementById('update-model-id').value = modelId;
            document.getElementById('update-model-name').value = model.model.name || '';
            document.getElementById('update-model-description').value = model.model.description || '';
            document.getElementById('update-model-version').value = model.model.version || '1.0.0';
            modal.classList.remove('hidden');
        } else {
            showNotification('Update modal not found', 'error');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        showNotification('Failed to load model: ' + (error.message || 'Unknown error'), 'error');
    }
}

function closeUpdateModelModal() {
    document.getElementById('update-model-modal').classList.add('hidden');
}

async function submitUpdateModel(e) {
    e.preventDefault();
    const modelId = document.getElementById('update-model-id').value;
    const updateData = {
        name: document.getElementById('update-model-name').value,
        description: document.getElementById('update-model-description').value,
        version: document.getElementById('update-model-version').value
    };
    
    try {
        await apiCall(`/models/${modelId}`, {
            method: 'PUT',
            body: JSON.stringify(updateData)
        });
        
        showNotification('Model updated successfully', 'success');
        closeUpdateModelModal();
        loadModels();
    } catch (error) {
        console.error('Error updating model:', error);
        showNotification('Failed to update model: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function validateModel(modelId) {
    try {
        const result = await apiCall(`/models/${modelId}/validate`, {
            method: 'POST'
        });
        
        showNotification(`Model validation: ${result.valid ? 'Valid' : 'Invalid'}`, result.valid ? 'success' : 'error');
        if (result.valid) {
            loadModels(); // Refresh to show updated status
        }
    } catch (error) {
        console.error('Error validating model:', error);
        showNotification('Failed to validate model: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getModelMetrics(modelId) {
    try {
        const metrics = await apiCall(`/models/${modelId}/metrics`);
        
        const metricsModal = document.getElementById('model-metrics-modal');
        if (metricsModal) {
            document.getElementById('model-metrics-content').innerHTML = `
                <div class="space-y-4">
                    <div class="grid grid-cols-3 gap-4">
                        <div class="bg-blue-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Total Predictions</div>
                            <div class="text-2xl font-bold text-blue-600">${metrics.total_predictions || 0}</div>
                        </div>
                        <div class="bg-green-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Success Rate</div>
                            <div class="text-2xl font-bold text-green-600">${((metrics.success_rate || 0) * 100).toFixed(1)}%</div>
                        </div>
                        <div class="bg-purple-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Avg Response Time</div>
                            <div class="text-2xl font-bold text-purple-600">${Math.round(metrics.avg_response_time_ms || 0)}ms</div>
                        </div>
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-800 mb-2">Full Metrics</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                    </div>
                </div>
            `;
            metricsModal.classList.remove('hidden');
        } else {
            showNotification(`Metrics: ${metrics.total_predictions || 0} predictions`, 'info');
        }
    } catch (error) {
        console.error('Error getting model metrics:', error);
        showNotification('Failed to get model metrics: ' + (error.message || 'Unknown error'), 'error');
    }
}

function closeModelMetricsModal() {
    document.getElementById('model-metrics-modal').classList.add('hidden');
}

// Deployments Management
async function loadDeployments() {
    try {
        const deployments = await apiCall('/deployments/');
        currentDeployments = deployments;
        renderDeploymentsList(deployments);
    } catch (error) {
        console.error('Error loading deployments:', error);
    }
}

function renderDeploymentsList(deployments) {
    const container = document.getElementById('deployments-list');
    
    if (deployments.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-rocket text-4xl mb-4"></i>
                <p>No deployments yet</p>
                <p class="text-sm mt-2">Deploy a model to create API endpoints</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = deployments.map(deployment => `
        <div class="border-b border-gray-200 py-4 last:border-b-0">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <h4 class="text-lg font-medium text-gray-900">${deployment.service_name || deployment.deployment_name || 'Unnamed Deployment'}</h4>
                    <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                        <span><i class="fas fa-server mr-1"></i>${deployment.framework || 'N/A'}</span>
                        <span><i class="fas fa-calendar mr-1"></i>${new Date(deployment.created_at).toLocaleDateString()}</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getDeploymentStatusClass(deployment.status)}">${deployment.status}</span>
                    </div>
                    ${deployment.endpoint_url ? `
                        <div class="mt-2">
                            <span class="text-xs text-gray-500">Endpoint:</span>
                            <code class="ml-1 text-xs bg-gray-100 px-2 py-1 rounded">${deployment.endpoint_url}</code>
                        </div>
                    ` : ''}
                    ${deployment.description ? `<p class="mt-2 text-sm text-gray-600">${deployment.description}</p>` : ''}
                </div>
                <div class="flex items-center space-x-2">
                    <button onclick="getDeploymentStatus('${deployment.id}')" class="text-purple-600 hover:text-purple-800" title="Status">
                        <i class="fas fa-info-circle"></i>
                    </button>
                    <button onclick="getDeploymentMetrics('${deployment.id}')" class="text-indigo-600 hover:text-indigo-800" title="Metrics">
                        <i class="fas fa-chart-bar"></i>
                    </button>
                    <button onclick="testDeploymentEndpoint('${deployment.id}')" class="text-blue-600 hover:text-blue-800" title="Test">
                        <i class="fas fa-vial"></i>
                    </button>
                    <button onclick="testDeployment('${deployment.id}')" class="text-blue-600 hover:text-blue-800" title="Quick Test">
                        <i class="fas fa-play"></i>
                    </button>
                    ${deployment.status === 'stopped' ? `
                        <button onclick="startDeployment('${deployment.id}')" class="text-green-600 hover:text-green-800" title="Start">
                            <i class="fas fa-play-circle"></i>
                        </button>
                    ` : `
                        <button onclick="stopDeployment('${deployment.id}')" class="text-red-600 hover:text-red-800" title="Stop">
                            <i class="fas fa-stop"></i>
                        </button>
                    `}
                    <button onclick="openUpdateDeploymentModal('${deployment.id}')" class="text-yellow-600 hover:text-yellow-800" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button onclick="viewDeploymentLogs('${deployment.id}')" class="text-gray-600 hover:text-gray-800" title="Logs">
                        <i class="fas fa-file-alt"></i>
                    </button>
                    <button onclick="deleteDeployment('${deployment.id}')" class="text-red-600 hover:text-red-800" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

function getDeploymentStatusClass(status) {
    switch(status) {
        case 'active': return 'bg-green-100 text-green-800';
        case 'deploying': return 'bg-yellow-100 text-yellow-800';
        case 'stopped': return 'bg-gray-100 text-gray-800';
        case 'error': return 'bg-red-100 text-red-800';
        default: return 'bg-gray-100 text-gray-800';
    }
}

function testDeployment(deploymentId) {
    // Switch to test tab and select this deployment
    showTab('test');
    const testSelect = document.getElementById('test-model-select');
    testSelect.value = deploymentId;
    loadTestInterface();
}

// Model Testing
async function loadTestModels() {
    try {
        const deployments = await apiCall('/deployments/');
        const activeDeployments = deployments.filter(d => d.status === 'active');
        
        const testSelect = document.getElementById('test-model-select');
        testSelect.innerHTML = '<option value="">Select a deployed model...</option>';
        
        activeDeployments.forEach(deployment => {
            const option = document.createElement('option');
            option.value = deployment.id;
            option.textContent = `${deployment.service_name} (${deployment.framework})`;
            testSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading test models:', error);
    }
}

async function loadTestInterface() {
    const testSelect = document.getElementById('test-model-select');
    const deploymentId = testSelect.value;
    
    if (!deploymentId) {
        document.getElementById('test-interface').classList.add('hidden');
        return;
    }
    
    try {
        // Get deployment schema
        const schema = await apiCall(`/predict/${deploymentId}/schema`);
        
        renderTestForm(schema);
        renderProbaForm(schema);
        document.getElementById('test-interface').classList.remove('hidden');
        setPredictionType('single'); // Reset to single prediction
        
    } catch (error) {
        console.error('Error loading test interface:', error);
        showNotification('Failed to load test interface', 'error');
    }
}

function renderProbaForm(schema) {
    const container = document.getElementById('proba-input-fields');
    
    if (!schema.input_schema || !schema.input_schema.fields) {
        container.innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <p>No input schema defined for this model</p>
                <p class="text-sm mt-1">You can still test with custom JSON data</p>
            </div>
            <textarea id="custom-json-input-proba" class="w-full border-gray-300 rounded-md mt-4" rows="6" 
                      placeholder='{"data": [1, 2, 3, 4]}'></textarea>
        `;
        return;
    }
    
    container.innerHTML = schema.input_schema.fields.map(field => {
        const inputType = getInputType(field.data_type);
        return `
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">
                    ${field.name}
                    ${field.required ? '<span class="text-red-500">*</span>' : ''}
                </label>
                <input type="${inputType}" 
                       name="${field.name}" 
                       class="w-full border-gray-300 rounded-md"
                       ${field.required ? 'required' : ''}
                       ${field.min_value !== undefined ? `min="${field.min_value}"` : ''}
                       ${field.max_value !== undefined ? `max="${field.max_value}"` : ''}
                       placeholder="${field.description || field.name}">
                ${field.description ? `<p class="text-xs text-gray-500 mt-1">${field.description}</p>` : ''}
            </div>
        `;
    }).join('');
}

function renderTestForm(schema) {
    const container = document.getElementById('test-input-fields');
    
    if (!schema.input_schema || !schema.input_schema.fields) {
        container.innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <p>No input schema defined for this model</p>
                <p class="text-sm mt-1">You can still test with custom JSON data</p>
            </div>
            <textarea id="custom-json-input" class="w-full border-gray-300 rounded-md" rows="6" 
                      placeholder='{"data": [1, 2, 3, 4]}'></textarea>
        `;
        return;
    }
    
    container.innerHTML = schema.input_schema.fields.map(field => {
        const inputType = getInputType(field.data_type);
        return `
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">
                    ${field.name}
                    ${field.required ? '<span class="text-red-500">*</span>' : ''}
                </label>
                <input type="${inputType}" 
                       name="${field.name}" 
                       class="w-full border-gray-300 rounded-md"
                       ${field.required ? 'required' : ''}
                       ${field.min_value !== undefined ? `min="${field.min_value}"` : ''}
                       ${field.max_value !== undefined ? `max="${field.max_value}"` : ''}
                       placeholder="${field.description || field.name}">
                ${field.description ? `<p class="text-xs text-gray-500 mt-1">${field.description}</p>` : ''}
            </div>
        `;
    }).join('');
}

function getInputType(dataType) {
    switch(dataType) {
        case 'integer': return 'number';
        case 'float': return 'number';
        case 'boolean': return 'checkbox';
        case 'date': return 'date';
        case 'datetime': return 'datetime-local';
        default: return 'text';
    }
}

// Prediction Type Management
let currentPredictionType = 'single';

function setPredictionType(type) {
    currentPredictionType = type;
    
    // Update button styles
    document.querySelectorAll('.prediction-type-btn').forEach(btn => {
        btn.classList.remove('bg-blue-600', 'text-white');
        btn.classList.add('bg-gray-200', 'text-gray-700');
    });
    document.getElementById(`pred-type-${type}`).classList.remove('bg-gray-200', 'text-gray-700');
    document.getElementById(`pred-type-${type}`).classList.add('bg-blue-600', 'text-white');
    
    // Show/hide forms
    document.querySelectorAll('.prediction-form').forEach(form => form.classList.add('hidden'));
    document.getElementById(`${type === 'single' ? 'test' : type === 'batch' ? 'batch-test' : 'proba-test'}-form`).classList.remove('hidden');
    
    // Clear results
    document.getElementById('test-results').innerHTML = `
        <div class="text-center py-8 text-gray-500">
            <i class="fas fa-chart-bar text-4xl mb-4"></i>
            <p>Run a test to see results</p>
        </div>
    `;
}

async function handleModelTest(e) {
    e.preventDefault();
    
    const testSelect = document.getElementById('test-model-select');
    const deploymentId = testSelect.value;
    
    if (!deploymentId) {
        showNotification('Please select a model to test', 'error');
        return;
    }
    
    try {
        // Collect test data
        let testData;
        const customJsonInput = document.getElementById('custom-json-input');
        
        if (customJsonInput) {
            // Use custom JSON input
            testData = JSON.parse(customJsonInput.value);
        } else {
            // Collect form data
            const formData = new FormData(e.target);
            testData = {};
            
            for (const [key, value] of formData.entries()) {
                testData[key] = value;
            }
        }
        
        // Make prediction
        const result = await apiCall(`/predict/${deploymentId}`, {
            method: 'POST',
            body: JSON.stringify(testData)
        });
        
        displayTestResults(result);
        
    } catch (error) {
        console.error('Test error:', error);
        document.getElementById('test-results').innerHTML = `
            <div class="text-center py-8 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p>Test failed</p>
                <p class="text-sm mt-2">${error.message}</p>
            </div>
        `;
    }
}

async function handleBatchTest(e) {
    e.preventDefault();
    
    const testSelect = document.getElementById('test-model-select');
    const deploymentId = testSelect.value;
    
    if (!deploymentId) {
        showNotification('Please select a model to test', 'error');
        return;
    }
    
    try {
        const batchData = JSON.parse(document.getElementById('batch-test-data').value);
        
        if (!Array.isArray(batchData)) {
            throw new Error('Batch data must be an array');
        }
        
        const result = await apiCall(`/predict/${deploymentId}/batch`, {
            method: 'POST',
            body: JSON.stringify({ data: batchData })
        });
        
        displayBatchTestResults(result);
        
    } catch (error) {
        console.error('Batch test error:', error);
        document.getElementById('test-results').innerHTML = `
            <div class="text-center py-8 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p>Batch test failed</p>
                <p class="text-sm mt-2">${error.message}</p>
            </div>
        `;
    }
}

function displayBatchTestResults(result) {
    const container = document.getElementById('test-results');
    
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 class="font-medium text-green-800 mb-2">
                    <i class="fas fa-check-circle mr-2"></i>Batch Prediction Successful
                </h4>
                <div class="text-sm text-green-700">
                    <p><strong>Total Predictions:</strong> ${result.predictions ? result.predictions.length : 0}</p>
                </div>
            </div>
            
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">Batch Results</h4>
                <div class="max-h-96 overflow-y-auto">
                    <table class="min-w-full text-xs">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="px-2 py-1 text-left">Index</th>
                                <th class="px-2 py-1 text-left">Prediction</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.predictions ? result.predictions.map((pred, idx) => `
                                <tr class="border-b">
                                    <td class="px-2 py-1">${idx + 1}</td>
                                    <td class="px-2 py-1 font-mono">${JSON.stringify(pred).substring(0, 100)}${JSON.stringify(pred).length > 100 ? '...' : ''}</td>
                                </tr>
                            `).join('') : '<tr><td colspan="2" class="px-2 py-1 text-center text-gray-500">No predictions</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">Full Response</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        </div>
    `;
}

async function handleProbaTest(e) {
    e.preventDefault();
    
    const testSelect = document.getElementById('test-model-select');
    const deploymentId = testSelect.value;
    
    if (!deploymentId) {
        showNotification('Please select a model to test', 'error');
        return;
    }
    
    try {
        // Collect test data
        let testData;
        const customJsonInput = document.getElementById('custom-json-input');
        
        if (customJsonInput && customJsonInput.value) {
            testData = JSON.parse(customJsonInput.value);
        } else {
            // Collect form data
            const formData = new FormData(e.target);
            testData = {};
            
            for (const [key, value] of formData.entries()) {
                testData[key] = value;
            }
        }
        
        // Make probability prediction
        const result = await apiCall(`/predict/${deploymentId}/proba`, {
            method: 'POST',
            body: JSON.stringify({ data: testData })
        });
        
        displayProbaTestResults(result);
        
    } catch (error) {
        console.error('Probability test error:', error);
        document.getElementById('test-results').innerHTML = `
            <div class="text-center py-8 text-red-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p>Probability test failed</p>
                <p class="text-sm mt-2">${error.message}</p>
            </div>
        `;
    }
}

function displayProbaTestResults(result) {
    const container = document.getElementById('test-results');
    
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 class="font-medium text-purple-800 mb-2">
                    <i class="fas fa-percentage mr-2"></i>Probability Prediction Successful
                </h4>
            </div>
            
            ${result.probabilities ? `
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Class Probabilities</h4>
                    <div class="space-y-2">
                        ${Array.isArray(result.probabilities) ? 
                            result.probabilities.map((prob, idx) => `
                                <div class="flex items-center justify-between bg-white p-2 rounded">
                                    <span class="text-sm">Class ${idx}:</span>
                                    <span class="font-mono text-sm">${(prob * 100).toFixed(2)}%</span>
                                </div>
                            `).join('') :
                            Object.entries(result.probabilities).map(([class_name, prob]) => `
                                <div class="flex items-center justify-between bg-white p-2 rounded">
                                    <span class="text-sm">${class_name}:</span>
                                    <span class="font-mono text-sm">${(prob * 100).toFixed(2)}%</span>
                                </div>
                            `).join('')
                        }
                    </div>
                </div>
            ` : ''}
            
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">Full Response</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        </div>
    `;
}

function displayTestResults(result) {
    const container = document.getElementById('test-results');
    
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 class="font-medium text-green-800 mb-2">
                    <i class="fas fa-check-circle mr-2"></i>Prediction Successful
                </h4>
                <div class="text-sm text-green-700">
                    <p><strong>Predictions:</strong></p>
                    <pre class="mt-2 bg-green-100 p-2 rounded text-xs overflow-x-auto">${JSON.stringify(result.predictions, null, 2)}</pre>
                </div>
            </div>
            
            ${result.validation ? `
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 class="font-medium text-blue-800 mb-2">
                    <i class="fas fa-shield-alt mr-2"></i>Validation Info
                </h4>
                <div class="text-sm text-blue-700">
                    <p><strong>Validation Performed:</strong> ${result.validation.validation_performed ? 'Yes' : 'No'}</p>
                    <p><strong>Message:</strong> ${result.validation.validation_message}</p>
                </div>
            </div>
            ` : ''}
            
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">
                    <i class="fas fa-info-circle mr-2"></i>Full Response
                </h4>
                <pre class="text-xs bg-gray-100 p-2 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        </div>
    `;
}

async function loadExampleData() {
    const testSelect = document.getElementById('test-model-select');
    const deploymentId = testSelect.value;
    
    if (!deploymentId) {
        showNotification('Please select a model first', 'error');
        return;
    }
    
    try {
        const schema = await apiCall(`/predict/${deploymentId}/schema`);
        
        if (schema.example_input) {
            // Load example data into form fields
            Object.keys(schema.example_input).forEach(fieldName => {
                const input = document.querySelector(`input[name="${fieldName}"]`);
                if (input) {
                    input.value = schema.example_input[fieldName];
                }
            });
            
            showNotification('Example data loaded', 'success');
        } else {
            showNotification('No example data available', 'warning');
        }
        
    } catch (error) {
        console.error('Error loading example data:', error);
    }
}

// Utility Functions
function viewModelDetails(modelId) {
    // This would open a modal or navigate to a detailed view
    console.log('View model details:', modelId);
    showNotification('Model details view not implemented yet', 'info');
}

function viewDeploymentLogs(deploymentId) {
    // This would show deployment logs
    console.log('View deployment logs:', deploymentId);
    showNotification('Deployment logs view not implemented yet', 'info');
}

// Deployment Management Functions
async function startDeployment(deploymentId) {
    try {
        await apiCall(`/deployments/${deploymentId}/start`, {
            method: 'POST'
        });
        showNotification('Deployment started successfully', 'success');
        loadDeployments();
    } catch (error) {
        console.error('Error starting deployment:', error);
        showNotification('Failed to start deployment: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function stopDeployment(deploymentId) {
    if (!confirm('Are you sure you want to stop this deployment?')) {
        return;
    }
    
    try {
        await apiCall(`/deployments/${deploymentId}/stop`, {
            method: 'POST'
        });
        
        showNotification('Deployment stopped', 'success');
        loadDeployments();
        
    } catch (error) {
        console.error('Error stopping deployment:', error);
        showNotification('Failed to stop deployment: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function deleteDeployment(deploymentId) {
    if (!confirm('Are you sure you want to delete this deployment? This action cannot be undone.')) {
        return;
    }
    
    try {
        await apiCall(`/deployments/${deploymentId}`, {
            method: 'DELETE'
        });
        
        showNotification('Deployment deleted successfully', 'success');
        loadDeployments();
        
    } catch (error) {
        console.error('Error deleting deployment:', error);
        showNotification('Failed to delete deployment: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getDeploymentStatus(deploymentId) {
    try {
        const status = await apiCall(`/deployments/${deploymentId}/status`);
        
        // Show status in a modal or notification
        const statusModal = document.getElementById('deployment-status-modal');
        if (statusModal) {
            document.getElementById('deployment-status-content').innerHTML = `
                <div class="space-y-4">
                    <div class="bg-${status.status === 'active' ? 'green' : 'yellow'}-50 border border-${status.status === 'active' ? 'green' : 'yellow'}-200 rounded-lg p-4">
                        <h4 class="font-medium text-${status.status === 'active' ? 'green' : 'yellow'}-800 mb-2">Status: ${status.status}</h4>
                        ${status.service_health ? `<p class="text-sm text-${status.status === 'active' ? 'green' : 'yellow'}-700">Service Health: ${status.service_health}</p>` : ''}
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-800 mb-2">Details</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(status, null, 2)}</pre>
                    </div>
                </div>
            `;
            statusModal.classList.remove('hidden');
        } else {
            showNotification(`Deployment Status: ${status.status}`, 'info');
        }
    } catch (error) {
        console.error('Error getting deployment status:', error);
        showNotification('Failed to get deployment status: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getDeploymentMetrics(deploymentId) {
    try {
        const metrics = await apiCall(`/deployments/${deploymentId}/metrics`);
        
        // Show metrics in a modal
        const metricsModal = document.getElementById('deployment-metrics-modal');
        if (metricsModal) {
            document.getElementById('deployment-metrics-content').innerHTML = `
                <div class="space-y-4">
                    <div class="grid grid-cols-3 gap-4">
                        <div class="bg-blue-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Total Requests</div>
                            <div class="text-2xl font-bold text-blue-600">${metrics.total_requests || 0}</div>
                        </div>
                        <div class="bg-green-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Success Rate</div>
                            <div class="text-2xl font-bold text-green-600">${((metrics.success_rate || 0) * 100).toFixed(1)}%</div>
                        </div>
                        <div class="bg-purple-50 p-4 rounded">
                            <div class="text-sm text-gray-600">Avg Latency</div>
                            <div class="text-2xl font-bold text-purple-600">${Math.round(metrics.avg_latency_ms || 0)}ms</div>
                        </div>
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-800 mb-2">Full Metrics</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                    </div>
                </div>
            `;
            metricsModal.classList.remove('hidden');
        } else {
            showNotification(`Metrics loaded: ${metrics.total_requests || 0} requests`, 'info');
        }
    } catch (error) {
        console.error('Error getting deployment metrics:', error);
        showNotification('Failed to get deployment metrics: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function testDeploymentEndpoint(deploymentId) {
    try {
        // Get deployment to show test interface
        const deployment = await apiCall(`/deployments/${deploymentId}`);
        
        // Open test modal with deployment info
        const testModal = document.getElementById('deployment-test-modal');
        if (testModal) {
            document.getElementById('deployment-test-deployment-id').value = deploymentId;
            document.getElementById('deployment-test-deployment-name').textContent = deployment.service_name || deployment.deployment_name || 'Deployment';
            testModal.classList.remove('hidden');
        } else {
            // Fallback: switch to test tab
            testDeployment(deploymentId);
        }
    } catch (error) {
        console.error('Error testing deployment:', error);
        showNotification('Failed to test deployment: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function submitDeploymentTest(e) {
    e.preventDefault();
    const deploymentId = document.getElementById('deployment-test-deployment-id').value;
    const testDataInput = document.getElementById('deployment-test-data').value;
    
    try {
        const testData = JSON.parse(testDataInput);
        const result = await apiCall(`/deployments/${deploymentId}/test`, {
            method: 'POST',
            body: JSON.stringify(testData)
        });
        
        document.getElementById('deployment-test-result').innerHTML = `
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 class="font-medium text-green-800 mb-2">Test Successful</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
    } catch (error) {
        document.getElementById('deployment-test-result').innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 class="font-medium text-red-800 mb-2">Test Failed</h4>
                <p class="text-sm text-red-700">${error.message || 'Unknown error'}</p>
            </div>
        `;
    }
}

function openUpdateDeploymentModal(deploymentId) {
    // Load deployment data and show update modal
    apiCall(`/deployments/${deploymentId}`).then(deployment => {
        const modal = document.getElementById('update-deployment-modal');
        if (modal) {
            document.getElementById('update-deployment-id').value = deploymentId;
            document.getElementById('update-deployment-name').value = deployment.deployment_name || deployment.service_name || '';
            document.getElementById('update-deployment-description').value = deployment.description || '';
            modal.classList.remove('hidden');
        } else {
            showNotification('Update modal not found', 'error');
        }
    }).catch(error => {
        console.error('Error loading deployment:', error);
        showNotification('Failed to load deployment: ' + (error.message || 'Unknown error'), 'error');
    });
}

function closeUpdateDeploymentModal() {
    document.getElementById('update-deployment-modal').classList.add('hidden');
}

async function submitUpdateDeployment(e) {
    e.preventDefault();
    const deploymentId = document.getElementById('update-deployment-id').value;
    const updateData = {
        deployment_name: document.getElementById('update-deployment-name').value,
        description: document.getElementById('update-deployment-description').value
    };
    
    try {
        await apiCall(`/deployments/${deploymentId}`, {
            method: 'PATCH',
            body: JSON.stringify(updateData)
        });
        
        showNotification('Deployment updated successfully', 'success');
        closeUpdateDeploymentModal();
        loadDeployments();
    } catch (error) {
        console.error('Error updating deployment:', error);
        showNotification('Failed to update deployment: ' + (error.message || 'Unknown error'), 'error');
    }
}

function closeDeploymentStatusModal() {
    document.getElementById('deployment-status-modal').classList.add('hidden');
}

function closeDeploymentMetricsModal() {
    document.getElementById('deployment-metrics-modal').classList.add('hidden');
}

function closeDeploymentTestModal() {
    document.getElementById('deployment-test-modal').classList.add('hidden');
    document.getElementById('deployment-test-result').innerHTML = '';
}

// Export functions for global access
window.showTab = showTab;
window.toggleSchemaConfig = toggleSchemaConfig;
window.addInputField = addInputField;
window.addOutputField = addOutputField;
window.removeField = removeField;
window.loadTemplate = loadTemplate;
window.openSaveTemplateModal = openSaveTemplateModal;
window.closeSaveTemplateModal = closeSaveTemplateModal;
window.saveTemplate = saveTemplate;
window.loadCustomTemplate = loadCustomTemplate;
window.deleteCustomTemplate = deleteCustomTemplate;
window.clearFile = clearFile;
window.loadTestInterface = loadTestInterface;
window.loadExampleData = loadExampleData;
window.viewModelDetails = viewModelDetails;
window.deployModel = deployModel;
window.deleteModel = deleteModel;
window.testDeployment = testDeployment;
window.viewDeploymentLogs = viewDeploymentLogs;
window.stopDeployment = stopDeployment;

// ============================================
// MONITORING FUNCTIONS
// ============================================

// Monitoring State
let currentMonitoringData = {
    health: null,
    alerts: [],
    drift: [],
    performance: {}
};

// Load Monitoring Data
async function loadMonitoringData() {
    try {
        // Load system health
        await loadSystemHealth();
        
        // Load active alerts
        await loadAlerts();
        
        // Load recent drift detections
        await loadRecentDrift();
        
        // Load performance metrics
        await loadPerformanceMetrics();
        
    } catch (error) {
        console.error('Error loading monitoring data:', error);
        showNotification('Failed to load monitoring data', 'error');
    }
}

// System Health
async function loadSystemHealth() {
    try {
        const health = await apiCall('/monitoring/health');
        currentMonitoringData.health = health;
        renderSystemHealth(health);
    } catch (error) {
        console.error('Error loading system health:', error);
    }
}

function renderSystemHealth(health) {
    const container = document.getElementById('system-health-display');
    if (!container) return;
    
    const statusClass = health.overall_status === 'operational' ? 'text-green-600' : 
                       health.overall_status === 'degraded' ? 'text-yellow-600' : 'text-red-600';
    
    container.innerHTML = `
        <div class="space-y-4">
            <div class="flex items-center justify-between">
                <h4 class="text-lg font-medium">System Status</h4>
                <span class="px-3 py-1 rounded-full text-sm font-medium ${statusClass}">
                    ${health.overall_status || 'unknown'}
                </span>
            </div>
            ${health.components ? `
                <div class="space-y-2">
                    ${health.components.map(component => `
                        <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span class="text-sm">${component.component}</span>
                            <span class="text-xs px-2 py-1 rounded ${component.status === 'operational' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                ${component.status}
                            </span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// Alerts
async function loadAlerts(activeOnly = true) {
    try {
        const alerts = await apiCall(`/monitoring/alerts?active_only=${activeOnly}&limit=10`);
        currentMonitoringData.alerts = alerts;
        renderAlerts(alerts);
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

function renderAlerts(alerts) {
    const container = document.getElementById('alerts-display');
    if (!container) return;
    
    if (alerts.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No active alerts</p>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => {
        const severityClass = alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                             alert.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                             'bg-blue-100 text-blue-800';
        return `
            <div class="border-l-4 ${alert.severity === 'critical' ? 'border-red-500' : 'border-yellow-500'} p-3 bg-white rounded mb-2">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <h5 class="font-medium text-gray-900">${alert.title}</h5>
                        <p class="text-sm text-gray-600 mt-1">${alert.description}</p>
                        <div class="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                            <span class="px-2 py-1 rounded ${severityClass}">${alert.severity}</span>
                            <span>${new Date(alert.triggered_at).toLocaleString()}</span>
                        </div>
                    </div>
                    ${alert.is_active ? `
                        <button onclick="resolveAlert('${alert.id}')" class="text-green-600 hover:text-green-800 text-sm">
                            <i class="fas fa-check"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
}

async function resolveAlert(alertId) {
    try {
        await apiCall(`/monitoring/alerts/${alertId}/resolve`, { method: 'POST' });
        showNotification('Alert resolved', 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error resolving alert:', error);
    }
}

// Drift Detection
async function loadRecentDrift() {
    try {
        const container = document.getElementById('drift-display');
        if (container) {
            container.innerHTML = '<p class="text-gray-500 text-center py-4">Select a model to view drift detection</p>';
        }
    } catch (error) {
        console.error('Error loading drift:', error);
    }
}

async function detectDrift(modelId, driftType = 'feature') {
    try {
        const endTime = new Date();
        const startTime = new Date(endTime.getTime() - 7 * 24 * 60 * 60 * 1000);
        
        const params = new URLSearchParams({
            baseline_window_start: startTime.toISOString(),
            baseline_window_end: new Date(startTime.getTime() + 24 * 60 * 60 * 1000).toISOString(),
            current_window_start: new Date(endTime.getTime() - 24 * 60 * 60 * 1000).toISOString(),
            current_window_end: endTime.toISOString()
        });
        
        const drift = await apiCall(`/monitoring/models/${modelId}/drift/${driftType}?${params}`, {
            method: 'POST'
        });
        
        showNotification(`${driftType} drift detection completed`, 'success');
        return drift;
    } catch (error) {
        console.error('Error detecting drift:', error);
        showNotification('Drift detection failed: ' + error.message, 'error');
    }
}

// Performance Metrics
async function loadPerformanceMetrics() {
    try {
        const container = document.getElementById('performance-metrics-display');
        if (container) {
            container.innerHTML = '<p class="text-gray-500 text-center py-4">Select a model to view performance metrics</p>';
        }
    } catch (error) {
        console.error('Error loading performance metrics:', error);
    }
}

async function loadModelPerformance(modelId, deploymentId = null) {
    try {
        const endTime = new Date();
        const startTime = new Date(endTime.getTime() - 24 * 60 * 60 * 1000);
        
        const params = new URLSearchParams({
            start_time: startTime.toISOString(),
            end_time: endTime.toISOString()
        });
        if (deploymentId) params.append('deployment_id', deploymentId);
        
        const metrics = await apiCall(`/monitoring/models/${modelId}/performance?${params}`);
        renderPerformanceMetrics(metrics);
        return metrics;
    } catch (error) {
        console.error('Error loading model performance:', error);
        showNotification('Failed to load performance metrics', 'error');
    }
}

function renderPerformanceMetrics(metrics) {
    const container = document.getElementById('performance-metrics-display');
    if (!container) return;
    
    container.innerHTML = `
        <div class="grid grid-cols-2 gap-4">
            <div class="bg-blue-50 p-4 rounded">
                <div class="text-sm text-gray-600">Total Predictions</div>
                <div class="text-2xl font-bold text-blue-600">${metrics.total_predictions || 0}</div>
            </div>
            <div class="bg-green-50 p-4 rounded">
                <div class="text-sm text-gray-600">Success Rate</div>
                <div class="text-2xl font-bold text-green-600">${((metrics.success_rate || 0) * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-purple-50 p-4 rounded">
                <div class="text-sm text-gray-600">Avg Latency</div>
                <div class="text-2xl font-bold text-purple-600">${Math.round(metrics.avg_latency_ms || 0)}ms</div>
            </div>
            <div class="bg-orange-50 p-4 rounded">
                <div class="text-sm text-gray-600">Error Rate</div>
                <div class="text-2xl font-bold text-orange-600">${((metrics.error_rate || 0) * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}

// Prediction Logs
async function loadPredictionLogs(modelId, limit = 50) {
    try {
        const logs = await apiCall(`/monitoring/models/${modelId}/predictions/logs?limit=${limit}`);
        return logs;
    } catch (error) {
        console.error('Error loading prediction logs:', error);
        return [];
    }
}

// Aggregated Metrics
async function loadAggregatedMetrics(modelId = null, timeRange = '24h') {
    try {
        const params = new URLSearchParams({ time_range: timeRange });
        const endpoint = modelId 
            ? `/monitoring/models/${modelId}/metrics/aggregated?${params}`
            : `/monitoring/models/all/metrics/aggregated?${params}`;
        const metrics = await apiCall(endpoint);
        return metrics;
    } catch (error) {
        console.error('Error loading aggregated metrics:', error);
        return {};
    }
}

// Deployment Summary
async function loadDeploymentSummary(deploymentId) {
    try {
        const summary = await apiCall(`/monitoring/deployments/${deploymentId}/summary`);
        return summary;
    } catch (error) {
        console.error('Error loading deployment summary:', error);
        return null;
    }
}

// Confidence Metrics
async function loadConfidenceMetrics(modelId, startTime, endTime, deploymentId = null) {
    try {
        const params = new URLSearchParams({
            start_time: startTime.toISOString(),
            end_time: endTime.toISOString()
        });
        if (deploymentId) params.append('deployment_id', deploymentId);
        
        const metrics = await apiCall(`/monitoring/models/${modelId}/confidence?${params}`);
        return metrics;
    } catch (error) {
        console.error('Error loading confidence metrics:', error);
        return null;
    }
}

// Resource Usage
async function loadResourceUsage(modelId, deploymentId = null) {
    try {
        const params = deploymentId ? `?deployment_id=${deploymentId}` : '';
        const usage = await apiCall(`/monitoring/models/${modelId}/resources${params}`);
        return usage;
    } catch (error) {
        console.error('Error loading resource usage:', error);
        return null;
    }
}

// Enhanced Monitoring Functions - Connect existing functions to UI
async function loadModelPerformanceForUI() {
    const modelId = document.getElementById('performance-model-select').value;
    if (!modelId) {
        document.getElementById('performance-metrics-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to view performance metrics</div>';
        return;
    }
    await loadModelPerformance(modelId);
}

async function loadDriftForModel() {
    const modelId = document.getElementById('drift-model-select').value;
    if (!modelId) {
        document.getElementById('drift-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to view drift detection</div>';
        return;
    }
    // Load existing drift data if available
    const container = document.getElementById('drift-display');
    container.innerHTML = `
        <div class="text-center py-4">
            <p class="text-gray-600 mb-4">Ready to detect drift for model: ${modelId}</p>
            <p class="text-sm text-gray-500">Select drift type and click "Detect" to run analysis</p>
        </div>
    `;
}

async function runDriftDetection() {
    const modelId = document.getElementById('drift-model-select').value;
    const driftType = document.getElementById('drift-type-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    const drift = await detectDrift(modelId, driftType);
    if (drift) {
        renderDriftResults(drift, driftType);
        // Also load drift history after detection
        loadDriftHistory(modelId);
    }
}

async function loadDriftHistory(modelId) {
    try {
        const history = await apiCall(`/monitoring/models/${modelId}/drift`);
        renderDriftHistory(history);
    } catch (error) {
        console.error('Error loading drift history:', error);
        // Don't show error if endpoint doesn't exist yet
    }
}

function renderDriftHistory(history) {
    const container = document.getElementById('drift-display');
    if (!history || history.length === 0) {
        return; // Keep existing drift results
    }
    
    // Add history section below current results
    const historyHTML = `
        <div class="mt-4 bg-gray-50 border border-gray-200 rounded-lg p-4">
            <h4 class="font-medium text-gray-800 mb-2">Drift History</h4>
            <div class="space-y-2 max-h-64 overflow-y-auto">
                ${history.map(drift => `
                    <div class="border rounded p-2 bg-white">
                        <div class="flex justify-between items-center">
                            <div>
                                <p class="text-sm font-medium">${drift.drift_type || 'Unknown'} Drift</p>
                                <p class="text-xs text-gray-500">${new Date(drift.detected_at || drift.created_at).toLocaleString()}</p>
                            </div>
                            <span class="px-2 py-1 rounded text-xs ${drift.drift_detected ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}">
                                ${drift.drift_detected ? 'Detected' : 'No Drift'}
                            </span>
                        </div>
                        ${drift.drift_score !== undefined ? `
                            <p class="text-xs text-gray-600 mt-1">Score: ${drift.drift_score.toFixed(3)}</p>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', historyHTML);
}

async function getFeatureImportance(modelId) {
    try {
        const importance = await apiCall(`/monitoring/models/${modelId}/explain/importance`);
        
        const container = document.getElementById('explainability-display');
        container.innerHTML = `
            <div class="space-y-4">
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-medium text-blue-800 mb-2">Global Feature Importance</h4>
                    <div class="space-y-2">
                        ${Object.entries(importance.feature_importance || importance).sort((a, b) => b[1] - a[1]).map(([feature, score]) => `
                            <div class="flex items-center justify-between bg-white p-2 rounded">
                                <span class="text-sm">${feature}</span>
                                <div class="flex items-center space-x-2">
                                    <div class="w-32 bg-gray-200 rounded-full h-2">
                                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${(Math.abs(score) * 100)}%"></div>
                                    </div>
                                    <span class="text-sm font-mono">${(score * 100).toFixed(2)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Data</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(importance, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error getting feature importance:', error);
        showNotification('Failed to get feature importance: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function loadResourceUsageForUI() {
    const modelId = document.getElementById('performance-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const usage = await loadResourceUsage(modelId);
        if (usage) {
            const container = document.getElementById('performance-metrics-display');
            const existingHTML = container.innerHTML;
            container.innerHTML = existingHTML + `
                <div class="mt-4 bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Resource Usage</h4>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="bg-blue-50 p-3 rounded">
                            <div class="text-xs text-gray-600">CPU Usage</div>
                            <div class="text-xl font-bold text-blue-600">${(usage.cpu_usage || 0).toFixed(1)}%</div>
                        </div>
                        <div class="bg-green-50 p-3 rounded">
                            <div class="text-xs text-gray-600">Memory Usage</div>
                            <div class="text-xl font-bold text-green-600">${(usage.memory_usage_mb || 0).toFixed(0)} MB</div>
                        </div>
                        <div class="bg-purple-50 p-3 rounded">
                            <div class="text-xs text-gray-600">GPU Usage</div>
                            <div class="text-xl font-bold text-purple-600">${(usage.gpu_usage || 0).toFixed(1)}%</div>
                        </div>
                    </div>
                    <div class="mt-2 bg-white p-3 rounded">
                        <pre class="text-xs overflow-x-auto">${JSON.stringify(usage, null, 2)}</pre>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading resource usage:', error);
    }
}

async function loadAggregatedMetricsForUI() {
    const modelId = document.getElementById('performance-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const metrics = await loadAggregatedMetrics(modelId, '24h');
        if (metrics) {
            const container = document.getElementById('performance-metrics-display');
            const existingHTML = container.innerHTML;
            container.innerHTML = existingHTML + `
                <div class="mt-4 bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                    <h4 class="font-medium text-indigo-800 mb-2">Aggregated Metrics (24h)</h4>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="text-xs text-gray-600">Total Requests</div>
                            <div class="text-xl font-bold text-indigo-600">${metrics.total_requests || 0}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-600">Avg Response Time</div>
                            <div class="text-xl font-bold text-indigo-600">${Math.round(metrics.avg_response_time_ms || 0)}ms</div>
                        </div>
                    </div>
                    <div class="mt-2 bg-white p-3 rounded">
                        <pre class="text-xs overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading aggregated metrics:', error);
    }
}

async function loadDeploymentSummaryForUI(deploymentId) {
    try {
        const summary = await loadDeploymentSummary(deploymentId);
        if (summary) {
            // Show in deployment details or modal
            showNotification(`Deployment Summary: ${summary.total_predictions || 0} predictions`, 'info');
        }
    } catch (error) {
        console.error('Error loading deployment summary:', error);
    }
}

async function loadConfidenceMetricsForUI() {
    const modelId = document.getElementById('performance-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const endTime = new Date();
        const startTime = new Date(endTime.getTime() - 24 * 60 * 60 * 1000);
        const metrics = await loadConfidenceMetrics(modelId, startTime, endTime);
        
        if (metrics) {
            const container = document.getElementById('performance-metrics-display');
            const existingHTML = container.innerHTML;
            container.innerHTML = existingHTML + `
                <div class="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 class="font-medium text-yellow-800 mb-2">Confidence Metrics (24h)</h4>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="text-xs text-gray-600">Avg Confidence</div>
                            <div class="text-xl font-bold text-yellow-600">${((metrics.avg_confidence || 0) * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-600">Low Confidence Count</div>
                            <div class="text-xl font-bold text-yellow-600">${metrics.low_confidence_count || 0}</div>
                        </div>
                    </div>
                    <div class="mt-2 bg-white p-3 rounded">
                        <pre class="text-xs overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading confidence metrics:', error);
    }
}

function renderDriftResults(drift, driftType) {
    const container = document.getElementById('drift-display');
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-${drift.drift_detected ? 'red' : 'green'}-50 border border-${drift.drift_detected ? 'red' : 'green'}-200 rounded-lg p-4">
                <h4 class="font-medium text-${drift.drift_detected ? 'red' : 'green'}-800 mb-2">
                    ${drift.drift_detected ? 'Drift Detected' : 'No Drift Detected'}
                </h4>
                <p class="text-sm text-${drift.drift_detected ? 'red' : 'green'}-700">
                    Drift Score: ${(drift.drift_score || 0).toFixed(3)}
                </p>
            </div>
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">Details</h4>
                <pre class="text-xs bg-gray-100 p-2 rounded overflow-x-auto">${JSON.stringify(drift, null, 2)}</pre>
            </div>
        </div>
    `;
}

async function loadPredictionLogsForUI() {
    const modelId = document.getElementById('logs-model-select').value;
    if (!modelId) {
        document.getElementById('prediction-logs-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to view prediction logs</div>';
        return;
    }
    const logs = await loadPredictionLogs(modelId);
    renderPredictionLogs(logs);
}

function renderPredictionLogs(logs) {
    const container = document.getElementById('prediction-logs-display');
    if (!logs || logs.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No prediction logs found</p>';
        return;
    }
    container.innerHTML = `
        <div class="space-y-2 max-h-96 overflow-y-auto">
            ${logs.map(log => `
                <div class="border rounded p-3 bg-gray-50">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <p class="text-sm font-medium">${new Date(log.timestamp || log.created_at).toLocaleString()}</p>
                            <p class="text-xs text-gray-600 mt-1">Model: ${log.model_id || 'N/A'}</p>
                            ${log.prediction ? `<p class="text-xs text-gray-500 mt-1">Prediction: ${JSON.stringify(log.prediction).substring(0, 100)}...</p>` : ''}
                        </div>
                        <span class="px-2 py-1 rounded text-xs ${log.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                            ${log.status || 'unknown'}
                        </span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Model Details View
async function viewModelDetails(modelId) {
    try {
        const model = await apiCall(`/models/${modelId}`);
        const modal = document.getElementById('model-details-modal');
        const content = document.getElementById('model-details-content');
        
        content.innerHTML = `
            <div class="space-y-4">
                <div>
                    <h4 class="text-lg font-medium text-gray-900">${model.name}</h4>
                    <p class="text-sm text-gray-600 mt-1">${model.description || 'No description'}</p>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm font-medium text-gray-700">Framework</p>
                        <p class="text-sm text-gray-900">${model.framework}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Model Type</p>
                        <p class="text-sm text-gray-900">${model.model_type}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Version</p>
                        <p class="text-sm text-gray-900">${model.version || '1.0.0'}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Status</p>
                        <p class="text-sm text-gray-900">${model.status}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Created</p>
                        <p class="text-sm text-gray-900">${new Date(model.created_at).toLocaleString()}</p>
                    </div>
                </div>
                ${model.input_schema || model.output_schema ? `
                    <div>
                        <h5 class="text-md font-medium text-gray-900 mb-2">Schemas</h5>
                        ${model.input_schema ? `
                            <div class="mb-2">
                                <p class="text-sm font-medium text-gray-700">Input Schema</p>
                                <pre class="text-xs bg-gray-100 p-2 rounded mt-1 overflow-x-auto">${JSON.stringify(model.input_schema, null, 2)}</pre>
                            </div>
                        ` : ''}
                        ${model.output_schema ? `
                            <div>
                                <p class="text-sm font-medium text-gray-700">Output Schema</p>
                                <pre class="text-xs bg-gray-100 p-2 rounded mt-1 overflow-x-auto">${JSON.stringify(model.output_schema, null, 2)}</pre>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
            </div>
        `;
        
        modal.classList.remove('hidden');
    } catch (error) {
        console.error('Error loading model details:', error);
        showNotification('Failed to load model details', 'error');
    }
}

function closeModelDetailsModal() {
    document.getElementById('model-details-modal').classList.add('hidden');
}

// Deployment Logs View
async function viewDeploymentLogs(deploymentId) {
    try {
        // Try to get logs from monitoring endpoint
        const deployment = await apiCall(`/deployments/${deploymentId}`);
        const modal = document.getElementById('deployment-logs-modal');
        const content = document.getElementById('deployment-logs-content');
        
        // Try to get prediction logs if model_id is available
        if (deployment.model_id) {
            try {
                const logs = await loadPredictionLogs(deployment.model_id, 100);
                if (logs && logs.length > 0) {
                    content.innerHTML = `
                        <div class="space-y-2">
                            ${logs.map(log => `
                                <div class="border-b pb-2">
                                    <div class="flex justify-between items-center mb-1">
                                        <span class="text-xs text-gray-600">${new Date(log.timestamp || log.created_at).toLocaleString()}</span>
                                        <span class="px-2 py-1 rounded text-xs ${log.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                            ${log.status || 'unknown'}
                                        </span>
                                    </div>
                                    <pre class="text-xs bg-gray-50 p-2 rounded overflow-x-auto">${JSON.stringify(log, null, 2)}</pre>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    content.innerHTML = '<p class="text-gray-500 text-center py-4">No logs available for this deployment</p>';
                }
            } catch (error) {
                content.innerHTML = `
                    <div class="space-y-2">
                        <div class="border-b pb-2">
                            <p class="text-sm font-medium mb-2">Deployment Information</p>
                            <pre class="text-xs bg-gray-50 p-2 rounded overflow-x-auto">${JSON.stringify(deployment, null, 2)}</pre>
                        </div>
                        <p class="text-gray-500 text-sm">Prediction logs not available</p>
                    </div>
                `;
            }
        } else {
            content.innerHTML = `
                <div class="border-b pb-2">
                    <p class="text-sm font-medium mb-2">Deployment Information</p>
                    <pre class="text-xs bg-gray-50 p-2 rounded overflow-x-auto">${JSON.stringify(deployment, null, 2)}</pre>
                </div>
            `;
        }
        
        modal.classList.remove('hidden');
    } catch (error) {
        console.error('Error loading deployment logs:', error);
        showNotification('Failed to load deployment logs', 'error');
    }
}

function closeDeploymentLogsModal() {
    document.getElementById('deployment-logs-modal').classList.add('hidden');
}

// Schema Management Functions
function showSchemaTab(tabName) {
    document.querySelectorAll('.schema-tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelectorAll('.schema-tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    
    const tabElement = document.getElementById(`schema-${tabName}`);
    if (tabElement) {
        tabElement.classList.remove('hidden');
    }
    
    if (event && event.target) {
        event.target.classList.remove('border-transparent', 'text-gray-500');
        event.target.classList.add('border-blue-500', 'text-blue-600');
    } else {
        // Find the button for this tab
        const buttons = document.querySelectorAll('.schema-tab-btn');
        buttons.forEach(btn => {
            if (btn.textContent.toLowerCase().includes(tabName)) {
                btn.classList.remove('border-transparent', 'text-gray-500');
                btn.classList.add('border-blue-500', 'text-blue-600');
            }
        });
    }
    
    // Load data when switching to manage or versions tabs
    if (tabName === 'manage') {
        populateSchemaModelSelector();
    } else if (tabName === 'versions') {
        // Versions tab is ready
    }
}

// Schema Validation
document.addEventListener('DOMContentLoaded', function() {
    // Batch test form
    const batchTestForm = document.getElementById('batch-test-form');
    if (batchTestForm) {
        batchTestForm.addEventListener('submit', handleBatchTest);
    }
    
    // Probability test form
    const probaTestForm = document.getElementById('proba-test-form');
    if (probaTestForm) {
        probaTestForm.addEventListener('submit', handleProbaTest);
    }
    
    const validateForm = document.getElementById('validate-schema-form');
    if (validateForm) {
        validateForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const schema = JSON.parse(document.getElementById('validate-schema-input').value);
                const data = JSON.parse(document.getElementById('validate-data-input').value);
                
                const result = await apiCall('/schemas/validate', {
                    method: 'POST',
                    body: JSON.stringify({ schema, data })
                });
                
                const resultDiv = document.getElementById('validate-result');
                resultDiv.innerHTML = `
                    <div class="mt-4 p-4 rounded-lg ${result.valid ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}">
                        <h4 class="font-medium ${result.valid ? 'text-green-800' : 'text-red-800'} mb-2">
                            ${result.valid ? '✓ Validation Passed' : '✗ Validation Failed'}
                        </h4>
                        ${result.errors && result.errors.length > 0 ? `
                            <ul class="list-disc list-inside text-sm ${result.valid ? 'text-green-700' : 'text-red-700'}">
                                ${result.errors.map(err => `<li>${err}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                `;
            } catch (error) {
                showNotification('Validation error: ' + error.message, 'error');
            }
        });
    }

    // Schema Generation
    const generateForm = document.getElementById('generate-schema-form');
    if (generateForm) {
        generateForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const sampleData = JSON.parse(document.getElementById('generate-data-input').value);
                const schemaType = document.getElementById('generate-schema-type').value;
                
                const result = await apiCall('/schemas/generate', {
                    method: 'POST',
                    body: JSON.stringify({ sample_data: sampleData, schema_type: schemaType })
                });
                
                const resultDiv = document.getElementById('generate-result');
                resultDiv.innerHTML = `
                    <div class="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <h4 class="font-medium text-green-800 mb-2">Schema Generated</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result.schema, null, 2)}</pre>
                    </div>
                `;
            } catch (error) {
                showNotification('Generation error: ' + error.message, 'error');
            }
        });
    }

    // Schema Comparison
    const compareForm = document.getElementById('compare-schema-form');
    if (compareForm) {
        compareForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const schema1 = JSON.parse(document.getElementById('compare-schema1-input').value);
                const schema2 = JSON.parse(document.getElementById('compare-schema2-input').value);
                const strict = document.getElementById('compare-strict').checked;
                
                const result = await apiCall('/schemas/compare', {
                    method: 'POST',
                    body: JSON.stringify({ schema1, schema2, strict_comparison: strict })
                });
                
                const resultDiv = document.getElementById('compare-result');
                resultDiv.innerHTML = `
                    <div class="mt-4 p-4 rounded-lg ${result.compatible ? 'bg-green-50 border border-green-200' : 'bg-yellow-50 border border-yellow-200'}">
                        <h4 class="font-medium ${result.compatible ? 'text-green-800' : 'text-yellow-800'} mb-2">
                            Compatibility: ${result.compatible ? 'Compatible' : 'Incompatible'}
                        </h4>
                        <p class="text-sm ${result.compatible ? 'text-green-700' : 'text-yellow-700'} mb-2">
                            Score: ${(result.compatibility_score || 0).toFixed(2)}
                        </p>
                        ${result.differences && result.differences.length > 0 ? `
                            <div class="mt-2">
                                <p class="text-sm font-medium mb-1">Differences:</p>
                                <ul class="list-disc list-inside text-sm">
                                    ${result.differences.map(diff => `<li>${JSON.stringify(diff)}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        ${result.breaking_changes && result.breaking_changes.length > 0 ? `
                            <div class="mt-2">
                                <p class="text-sm font-medium text-red-700 mb-1">Breaking Changes:</p>
                                <ul class="list-disc list-inside text-sm text-red-700">
                                    ${result.breaking_changes.map(change => `<li>${JSON.stringify(change)}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                `;
            } catch (error) {
                showNotification('Comparison error: ' + error.message, 'error');
            }
        });
    }

    // Schema Conversion
    const convertForm = document.getElementById('convert-schema-form');
    if (convertForm) {
        convertForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const sourceSchema = JSON.parse(document.getElementById('convert-source-input').value);
                const targetFormat = document.getElementById('convert-target-format').value;
                
                const result = await apiCall('/schemas/convert', {
                    method: 'POST',
                    body: JSON.stringify({ source_schema: sourceSchema, target_format: targetFormat })
                });
                
                const resultDiv = document.getElementById('convert-result');
                resultDiv.innerHTML = `
                    <div class="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                        <h4 class="font-medium text-green-800 mb-2">Schema Converted</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result.converted_schema || result, null, 2)}</pre>
                    </div>
                `;
            } catch (error) {
                showNotification('Conversion error: ' + error.message, 'error');
            }
        });
    }
    
    // Populate schema model selector when schemas tab is shown
    if (document.getElementById('schemas')) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const schemasTab = document.getElementById('schemas');
                    if (schemasTab && schemasTab.classList.contains('active')) {
                        populateSchemaModelSelector();
                    }
                }
            });
        });
        observer.observe(document.getElementById('schemas'), { attributes: true });
    }
});

// A/B Testing Functions
let currentABTests = [];
let selectedABTestId = null;

async function loadABTests() {
    try {
        // Try to get list of A/B tests (if endpoint exists)
        try {
            const tests = await apiCall('/monitoring/ab-tests');
            currentABTests = Array.isArray(tests) ? tests : [];
            renderABTestsList(currentABTests);
        } catch (error) {
            // If GET endpoint doesn't exist, show placeholder
            const container = document.getElementById('ab-tests-list');
            container.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <p class="mb-2">A/B test listing requires backend GET /monitoring/ab-tests endpoint</p>
                    <p class="text-sm">Use the "Create A/B Test" button to create a new test</p>
                    <p class="text-xs mt-2 text-gray-400">Note: Once created, tests will appear here when GET endpoint is available</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading A/B tests:', error);
    }
}

function renderABTestsList(tests) {
    const container = document.getElementById('ab-tests-list');
    
    if (!tests || tests.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>No A/B tests created yet</p>
                <p class="text-sm mt-2">Use the "Create A/B Test" button to create a new test</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = tests.map(test => `
        <div class="border-b border-gray-200 py-4 last:border-b-0">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <h4 class="text-lg font-medium text-gray-900">${test.test_name || 'Unnamed Test'}</h4>
                    <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                        <span><i class="fas fa-calendar mr-1"></i>${new Date(test.created_at || test.scheduled_start).toLocaleDateString()}</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getABTestStatusClass(test.status)}">${test.status || 'unknown'}</span>
                        <span>Variant A: ${test.variant_a_percentage || 50}%</span>
                        <span>Variant B: ${test.variant_b_percentage || 50}%</span>
                    </div>
                    ${test.description ? `<p class="mt-2 text-sm text-gray-600">${test.description}</p>` : ''}
                </div>
                <div class="flex items-center space-x-2">
                    <button onclick="getABTestDetails('${test.id || test.test_id}')" class="text-blue-600 hover:text-blue-800" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="getABTestMetrics('${test.id || test.test_id}')" class="text-purple-600 hover:text-purple-800" title="Metrics">
                        <i class="fas fa-chart-bar"></i>
                    </button>
                    ${test.status === 'running' ? `
                        <button onclick="stopABTest('${test.id || test.test_id}')" class="text-red-600 hover:text-red-800" title="Stop">
                            <i class="fas fa-stop"></i>
                        </button>
                    ` : `
                        <button onclick="startABTest('${test.id || test.test_id}')" class="text-green-600 hover:text-green-800" title="Start">
                            <i class="fas fa-play"></i>
                        </button>
                    `}
                </div>
            </div>
        </div>
    `).join('');
}

function getABTestStatusClass(status) {
    switch(status) {
        case 'running': return 'bg-green-100 text-green-800';
        case 'stopped': return 'bg-gray-100 text-gray-800';
        case 'completed': return 'bg-blue-100 text-blue-800';
        case 'error': return 'bg-red-100 text-red-800';
        default: return 'bg-yellow-100 text-yellow-800';
    }
}

async function getABTestDetails(testId) {
    try {
        // Try to get test details (if endpoint exists)
        try {
            const test = await apiCall(`/monitoring/ab-tests/${testId}`);
            showABTestDetails(test);
        } catch (error) {
            // If endpoint doesn't exist, show notification
            showNotification('GET endpoint for A/B test details not available', 'info');
        }
    } catch (error) {
        console.error('Error getting A/B test details:', error);
        showNotification('Failed to get A/B test details: ' + (error.message || 'Unknown error'), 'error');
    }
}

function showABTestDetails(test) {
    const modal = document.getElementById('ab-test-details-modal');
    if (modal) {
        document.getElementById('ab-test-details-content').innerHTML = `
            <div class="space-y-4">
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-medium text-blue-800 mb-2">${test.test_name}</h4>
                    <p class="text-sm text-blue-700">${test.description || 'No description'}</p>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm font-medium text-gray-700">Status</p>
                        <p class="text-sm text-gray-900">${test.status || 'unknown'}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Primary Metric</p>
                        <p class="text-sm text-gray-900">${test.primary_metric || 'N/A'}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Variant A</p>
                        <p class="text-sm text-gray-900">${test.variant_a_percentage || 50}%</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Variant B</p>
                        <p class="text-sm text-gray-900">${test.variant_b_percentage || 50}%</p>
                    </div>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Details</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(test, null, 2)}</pre>
                </div>
            </div>
        `;
        modal.classList.remove('hidden');
    }
}

function closeABTestDetailsModal() {
    document.getElementById('ab-test-details-modal').classList.add('hidden');
}

async function startABTest(testId) {
    try {
        await apiCall(`/monitoring/ab-tests/${testId}/start`, {
            method: 'POST'
        });
        showNotification('A/B test started successfully', 'success');
        loadABTests();
    } catch (error) {
        console.error('Error starting A/B test:', error);
        showNotification('Failed to start A/B test: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function stopABTest(testId) {
    if (!confirm('Are you sure you want to stop this A/B test?')) {
        return;
    }
    
    try {
        await apiCall(`/monitoring/ab-tests/${testId}/stop`, {
            method: 'POST'
        });
        showNotification('A/B test stopped successfully', 'success');
        loadABTests();
    } catch (error) {
        console.error('Error stopping A/B test:', error);
        showNotification('Failed to stop A/B test: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getABTestMetrics(testId) {
    try {
        const metrics = await apiCall(`/monitoring/ab-tests/${testId}/metrics`);
        
        selectedABTestId = testId;
        const metricsSection = document.getElementById('ab-test-metrics-section');
        const metricsDisplay = document.getElementById('ab-test-metrics-display');
        
        metricsSection.classList.remove('hidden');
        metricsDisplay.innerHTML = `
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-blue-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Variant A Requests</div>
                        <div class="text-2xl font-bold text-blue-600">${metrics.variant_a_requests || 0}</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Variant B Requests</div>
                        <div class="text-2xl font-bold text-green-600">${metrics.variant_b_requests || 0}</div>
                    </div>
                    <div class="bg-purple-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Variant A Performance</div>
                        <div class="text-2xl font-bold text-purple-600">${(metrics.variant_a_performance || 0).toFixed(2)}</div>
                    </div>
                    <div class="bg-orange-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Variant B Performance</div>
                        <div class="text-2xl font-bold text-orange-600">${(metrics.variant_b_performance || 0).toFixed(2)}</div>
                    </div>
                </div>
                ${metrics.statistical_significance !== undefined ? `
                    <div class="bg-${metrics.statistical_significance ? 'green' : 'yellow'}-50 border border-${metrics.statistical_significance ? 'green' : 'yellow'}-200 rounded-lg p-4">
                        <h4 class="font-medium text-${metrics.statistical_significance ? 'green' : 'yellow'}-800 mb-2">
                            Statistical Significance: ${metrics.statistical_significance ? 'Significant' : 'Not Significant'}
                        </h4>
                        <p class="text-sm text-${metrics.statistical_significance ? 'green' : 'yellow'}-700">
                            P-value: ${(metrics.p_value || 0).toFixed(4)}
                        </p>
                    </div>
                ` : ''}
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Metrics</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error getting A/B test metrics:', error);
        showNotification('Failed to get A/B test metrics: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function assignABTestVariant(testId, variant) {
    try {
        const result = await apiCall(`/monitoring/ab-tests/${testId}/assign`, {
            method: 'POST',
            body: JSON.stringify({ variant: variant })
        });
        
        showNotification(`Assigned to variant ${result.variant || variant}`, 'success');
    } catch (error) {
        console.error('Error assigning A/B test variant:', error);
        showNotification('Failed to assign variant: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateABTestModal() {
    document.getElementById('create-ab-test-modal').classList.remove('hidden');
    // Load models for selection
    loadModelsForABTest();
}

function closeCreateABTestModal() {
    document.getElementById('create-ab-test-modal').classList.add('hidden');
    document.getElementById('create-ab-test-form').reset();
}

async function loadModelsForABTest() {
    try {
        const models = await apiCall('/models/');
        const variantA = document.getElementById('ab-test-variant-a');
        const variantB = document.getElementById('ab-test-variant-b');
        
        variantA.innerHTML = '<option value="">Select model...</option>';
        variantB.innerHTML = '<option value="">Select model...</option>';
        
        models.forEach(model => {
            const optionA = document.createElement('option');
            optionA.value = model.id;
            optionA.textContent = `${model.name} (${model.framework})`;
            variantA.appendChild(optionA);
            
            const optionB = document.createElement('option');
            optionB.value = model.id;
            optionB.textContent = `${model.name} (${model.framework})`;
            variantB.appendChild(optionB);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const abTestForm = document.getElementById('create-ab-test-form');
    if (abTestForm) {
        abTestForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const testData = {
                    test_name: document.getElementById('ab-test-name').value,
                    description: document.getElementById('ab-test-description').value,
                    variant_a_model_id: document.getElementById('ab-test-variant-a').value,
                    variant_b_model_id: document.getElementById('ab-test-variant-b').value,
                    variant_a_percentage: parseInt(document.getElementById('ab-test-percent-a').value),
                    variant_b_percentage: parseInt(document.getElementById('ab-test-percent-b').value),
                    primary_metric: 'accuracy',
                    use_sticky_sessions: true,
                    min_sample_size: 1000,
                    significance_level: 0.05
                };
                
                const result = await apiCall('/monitoring/ab-tests', {
                    method: 'POST',
                    body: JSON.stringify(testData)
                });
                
                showNotification('A/B test created successfully!', 'success');
                closeCreateABTestModal();
                // Refresh the list (will work once GET endpoint is available)
                setTimeout(() => loadABTests(), 1000);
            } catch (error) {
                showNotification('Error creating A/B test: ' + error.message, 'error');
            }
        });
    }
});

// Canary Deployment Functions
function openCreateCanaryModal() {
    document.getElementById('create-canary-modal').classList.remove('hidden');
    loadModelsForCanary();
}

function closeCreateCanaryModal() {
    document.getElementById('create-canary-modal').classList.add('hidden');
    document.getElementById('create-canary-form').reset();
}

async function loadModelsForCanary() {
    try {
        const models = await apiCall('/models/');
        const select = document.getElementById('canary-model');
        select.innerHTML = '<option value="">Select model...</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.framework})`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const canaryForm = document.getElementById('create-canary-form');
    if (canaryForm) {
        canaryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const stages = document.getElementById('canary-stages').value.split(',').map(s => parseInt(s.trim()));
                const canaryData = {
                    deployment_name: document.getElementById('canary-name').value,
                    model_id: document.getElementById('canary-model').value,
                    initial_traffic_percentage: parseInt(document.getElementById('canary-traffic').value),
                    rollout_stages: stages,
                    health_check_enabled: true,
                    auto_rollback_enabled: true
                };
                
                const result = await apiCall('/monitoring/canary', {
                    method: 'POST',
                    body: JSON.stringify(canaryData)
                });
                
                showNotification('Canary deployment created successfully!', 'success');
                closeCreateCanaryModal();
                // Refresh the list (will work once GET endpoint is available)
                setTimeout(() => loadCanaryDeployments(), 1000);
            } catch (error) {
                showNotification('Error creating canary: ' + error.message, 'error');
            }
        });
    }
});

// Canary Deployment Management
let currentCanaries = [];

async function loadCanaryDeployments() {
    try {
        // Try to get list of canaries (if endpoint exists)
        try {
            const canaries = await apiCall('/monitoring/canary');
            currentCanaries = Array.isArray(canaries) ? canaries : [];
            renderCanaryList(currentCanaries);
        } catch (error) {
            // If GET endpoint doesn't exist, show placeholder
            const container = document.getElementById('canary-list');
            container.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <p class="mb-2">Canary listing requires backend GET /monitoring/canary endpoint</p>
                    <p class="text-sm">Use the "Create Canary" button to create a new canary deployment</p>
                    <p class="text-xs mt-2 text-gray-400">Note: Once created, canaries will appear here when GET endpoint is available</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading canary deployments:', error);
    }
}

function renderCanaryList(canaries) {
    const container = document.getElementById('canary-list');
    
    if (!canaries || canaries.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>No canary deployments yet</p>
                <p class="text-sm mt-2">Use the "Create Canary" button to create a new canary deployment</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = canaries.map(canary => `
        <div class="border-b border-gray-200 py-4 last:border-b-0">
            <div class="flex items-center justify-between">
                <div class="flex-1">
                    <h4 class="text-lg font-medium text-gray-900">${canary.deployment_name || 'Unnamed Canary'}</h4>
                    <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                        <span><i class="fas fa-calendar mr-1"></i>${new Date(canary.created_at || canary.scheduled_start).toLocaleDateString()}</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getCanaryStatusClass(canary.status)}">${canary.status || 'unknown'}</span>
                        <span>Traffic: ${canary.current_traffic_percentage || 0}%</span>
                        <span>Stage: ${canary.current_stage || 0}/${(canary.rollout_stages || []).length}</span>
                    </div>
                    ${canary.description ? `<p class="mt-2 text-sm text-gray-600">${canary.description}</p>` : ''}
                </div>
                <div class="flex items-center space-x-2">
                    <button onclick="getCanaryDetails('${canary.id || canary.canary_id}')" class="text-blue-600 hover:text-blue-800" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="getCanaryMetrics('${canary.id || canary.canary_id}')" class="text-purple-600 hover:text-purple-800" title="Metrics">
                        <i class="fas fa-chart-bar"></i>
                    </button>
                    <button onclick="checkCanaryHealth('${canary.id || canary.canary_id}')" class="text-green-600 hover:text-green-800" title="Health">
                        <i class="fas fa-heartbeat"></i>
                    </button>
                    ${canary.status === 'running' ? `
                        <button onclick="advanceCanaryRollout('${canary.id || canary.canary_id}')" class="text-blue-600 hover:text-blue-800" title="Advance">
                            <i class="fas fa-forward"></i>
                        </button>
                        <button onclick="rollbackCanary('${canary.id || canary.canary_id}')" class="text-red-600 hover:text-red-800" title="Rollback">
                            <i class="fas fa-undo"></i>
                        </button>
                    ` : canary.status === 'stopped' ? `
                        <button onclick="startCanaryRollout('${canary.id || canary.canary_id}')" class="text-green-600 hover:text-green-800" title="Start">
                            <i class="fas fa-play"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        </div>
    `).join('');
}

function getCanaryStatusClass(status) {
    switch(status) {
        case 'running': return 'bg-green-100 text-green-800';
        case 'stopped': return 'bg-gray-100 text-gray-800';
        case 'completed': return 'bg-blue-100 text-blue-800';
        case 'rolled_back': return 'bg-red-100 text-red-800';
        case 'error': return 'bg-red-100 text-red-800';
        default: return 'bg-yellow-100 text-yellow-800';
    }
}

async function getCanaryDetails(canaryId) {
    try {
        // Try to get canary details (if endpoint exists)
        try {
            const canary = await apiCall(`/monitoring/canary/${canaryId}`);
            showCanaryDetails(canary);
        } catch (error) {
            showNotification('GET endpoint for canary details not available', 'info');
        }
    } catch (error) {
        console.error('Error getting canary details:', error);
        showNotification('Failed to get canary details: ' + (error.message || 'Unknown error'), 'error');
    }
}

function showCanaryDetails(canary) {
    const modal = document.getElementById('canary-details-modal');
    if (modal) {
        document.getElementById('canary-details-content').innerHTML = `
            <div class="space-y-4">
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-medium text-blue-800 mb-2">${canary.deployment_name}</h4>
                    <p class="text-sm text-blue-700">${canary.description || 'No description'}</p>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm font-medium text-gray-700">Status</p>
                        <p class="text-sm text-gray-900">${canary.status || 'unknown'}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Current Traffic</p>
                        <p class="text-sm text-gray-900">${canary.current_traffic_percentage || 0}%</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Current Stage</p>
                        <p class="text-sm text-gray-900">${canary.current_stage || 0}/${(canary.rollout_stages || []).length}</p>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">Model ID</p>
                        <p class="text-sm text-gray-900">${canary.model_id || 'N/A'}</p>
                    </div>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Details</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(canary, null, 2)}</pre>
                </div>
            </div>
        `;
        modal.classList.remove('hidden');
    }
}

function closeCanaryDetailsModal() {
    document.getElementById('canary-details-modal').classList.add('hidden');
}

async function startCanaryRollout(canaryId) {
    try {
        await apiCall(`/monitoring/canary/${canaryId}/start`, {
            method: 'POST'
        });
        showNotification('Canary rollout started successfully', 'success');
        loadCanaryDeployments();
    } catch (error) {
        console.error('Error starting canary rollout:', error);
        showNotification('Failed to start canary rollout: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function advanceCanaryRollout(canaryId) {
    if (!confirm('Advance canary rollout to the next stage?')) {
        return;
    }
    
    try {
        await apiCall(`/monitoring/canary/${canaryId}/advance`, {
            method: 'POST'
        });
        showNotification('Canary rollout advanced successfully', 'success');
        loadCanaryDeployments();
    } catch (error) {
        console.error('Error advancing canary rollout:', error);
        showNotification('Failed to advance canary rollout: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function rollbackCanary(canaryId) {
    if (!confirm('Are you sure you want to rollback this canary deployment?')) {
        return;
    }
    
    try {
        await apiCall(`/monitoring/canary/${canaryId}/rollback`, {
            method: 'POST'
        });
        showNotification('Canary rolled back successfully', 'success');
        loadCanaryDeployments();
    } catch (error) {
        console.error('Error rolling back canary:', error);
        showNotification('Failed to rollback canary: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getCanaryMetrics(canaryId) {
    try {
        const metrics = await apiCall(`/monitoring/canary/${canaryId}/metrics`);
        
        selectedCanaryId = canaryId;
        const metricsSection = document.getElementById('canary-metrics-section');
        const metricsDisplay = document.getElementById('canary-metrics-display');
        
        metricsSection.classList.remove('hidden');
        metricsDisplay.innerHTML = `
            <div class="space-y-4">
                <div class="grid grid-cols-3 gap-4">
                    <div class="bg-blue-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Total Requests</div>
                        <div class="text-2xl font-bold text-blue-600">${metrics.total_requests || 0}</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Success Rate</div>
                        <div class="text-2xl font-bold text-green-600">${((metrics.success_rate || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="bg-purple-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Avg Latency</div>
                        <div class="text-2xl font-bold text-purple-600">${Math.round(metrics.avg_latency_ms || 0)}ms</div>
                    </div>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Metrics</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(metrics, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error getting canary metrics:', error);
        showNotification('Failed to get canary metrics: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function checkCanaryHealth(canaryId) {
    try {
        const health = await apiCall(`/monitoring/canary/${canaryId}/health`);
        
        const healthModal = document.getElementById('canary-health-modal');
        if (healthModal) {
            document.getElementById('canary-health-content').innerHTML = `
                <div class="space-y-4">
                    <div class="bg-${health.healthy ? 'green' : 'red'}-50 border border-${health.healthy ? 'green' : 'red'}-200 rounded-lg p-4">
                        <h4 class="font-medium text-${health.healthy ? 'green' : 'red'}-800 mb-2">
                            Health Status: ${health.healthy ? 'Healthy' : 'Unhealthy'}
                        </h4>
                        ${health.health_score !== undefined ? `
                            <p class="text-sm text-${health.healthy ? 'green' : 'red'}-700">
                                Health Score: ${(health.health_score * 100).toFixed(1)}%
                            </p>
                        ` : ''}
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-medium text-gray-800 mb-2">Health Details</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(health, null, 2)}</pre>
                    </div>
                </div>
            `;
            healthModal.classList.remove('hidden');
        } else {
            showNotification(`Canary Health: ${health.healthy ? 'Healthy' : 'Unhealthy'}`, health.healthy ? 'success' : 'error');
        }
    } catch (error) {
        console.error('Error checking canary health:', error);
        showNotification('Failed to check canary health: ' + (error.message || 'Unknown error'), 'error');
    }
}

function closeCanaryHealthModal() {
    document.getElementById('canary-health-modal').classList.add('hidden');
}

let selectedCanaryId = null;

// Governance Functions
function showGovernanceTab(tabName) {
    document.querySelectorAll('.governance-tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelectorAll('.governance-tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    
    document.getElementById(`governance-${tabName}`).classList.remove('hidden');
    event.target.classList.remove('border-transparent', 'text-gray-500');
    event.target.classList.add('border-blue-500', 'text-blue-600');
}

// Governance Functions
function openCreateLineageModal() {
    const modal = document.getElementById('create-lineage-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Lineage modal not found', 'error');
    }
}

function closeCreateLineageModal() {
    document.getElementById('create-lineage-modal').classList.add('hidden');
}

async function submitCreateLineage(e) {
    e.preventDefault();
    const lineageData = {
        lineage_type: document.getElementById('lineage-type').value,
        source_id: document.getElementById('lineage-source-id').value,
        source_type: document.getElementById('lineage-source-type').value,
        target_id: document.getElementById('lineage-target-id').value,
        target_type: document.getElementById('lineage-target-type').value,
        relationship_type: document.getElementById('lineage-relationship-type').value,
        source_metadata: JSON.parse(document.getElementById('lineage-source-metadata').value || '{}'),
        target_metadata: JSON.parse(document.getElementById('lineage-target-metadata').value || '{}'),
        relationship_metadata: JSON.parse(document.getElementById('lineage-relationship-metadata').value || '{}')
    };
    
    try {
        await apiCall('/monitoring/governance/lineage', {
            method: 'POST',
            body: JSON.stringify(lineageData)
        });
        
        showNotification('Data lineage created successfully', 'success');
        closeCreateLineageModal();
    } catch (error) {
        console.error('Error creating data lineage:', error);
        showNotification('Failed to create data lineage: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateWorkflowModal() {
    const modal = document.getElementById('create-workflow-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Workflow modal not found', 'error');
    }
}

function closeCreateWorkflowModal() {
    document.getElementById('create-workflow-modal').classList.add('hidden');
}

async function submitCreateWorkflow(e) {
    e.preventDefault();
    const workflowData = {
        workflow_type: document.getElementById('workflow-type').value,
        resource_type: document.getElementById('workflow-resource-type').value,
        resource_id: document.getElementById('workflow-resource-id').value,
        request_data: JSON.parse(document.getElementById('workflow-request-data').value || '{}'),
        policy_checks: JSON.parse(document.getElementById('workflow-policy-checks').value || '[]')
    };
    
    try {
        await apiCall('/monitoring/governance/workflows', {
            method: 'POST',
            body: JSON.stringify(workflowData)
        });
        
        showNotification('Governance workflow created successfully', 'success');
        closeCreateWorkflowModal();
    } catch (error) {
        console.error('Error creating workflow:', error);
        showNotification('Failed to create workflow: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateComplianceModal() {
    const modal = document.getElementById('create-compliance-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Compliance modal not found', 'error');
    }
}

function closeCreateComplianceModal() {
    document.getElementById('create-compliance-modal').classList.add('hidden');
}

async function submitCreateCompliance(e) {
    e.preventDefault();
    const complianceData = {
        compliance_type: document.getElementById('compliance-type').value,
        record_type: document.getElementById('compliance-record-type').value,
        subject_id: document.getElementById('compliance-subject-id').value,
        subject_type: document.getElementById('compliance-subject-type').value,
        description: document.getElementById('compliance-description').value,
        request_id: document.getElementById('compliance-request-id').value,
        requested_by: document.getElementById('compliance-requested-by').value,
        data_scope: document.getElementById('compliance-data-scope').value,
        additional_data: JSON.parse(document.getElementById('compliance-additional-data').value || '{}')
    };
    
    try {
        await apiCall('/monitoring/governance/compliance', {
            method: 'POST',
            body: JSON.stringify(complianceData)
        });
        
        showNotification('Compliance record created successfully', 'success');
        closeCreateComplianceModal();
    } catch (error) {
        console.error('Error creating compliance record:', error);
        showNotification('Failed to create compliance record: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateRetentionModal() {
    const modal = document.getElementById('create-retention-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Retention policy modal not found', 'error');
    }
}

function closeCreateRetentionModal() {
    document.getElementById('create-retention-modal').classList.add('hidden');
}

async function submitCreateRetentionPolicy(e) {
    e.preventDefault();
    const policyData = {
        policy_name: document.getElementById('retention-policy-name').value,
        policy_description: document.getElementById('retention-policy-description').value,
        resource_type: document.getElementById('retention-resource-type').value,
        model_id: document.getElementById('retention-model-id').value || null,
        deployment_id: document.getElementById('retention-deployment-id').value || null,
        retention_period_days: parseInt(document.getElementById('retention-period-days').value),
        retention_condition: document.getElementById('retention-condition').value,
        action_on_expiry: document.getElementById('retention-action').value,
        archive_location: document.getElementById('retention-archive-location').value,
        created_by: document.getElementById('retention-created-by').value
    };
    
    try {
        await apiCall('/monitoring/governance/retention-policies', {
            method: 'POST',
            body: JSON.stringify(policyData)
        });
        
        showNotification('Data retention policy created successfully', 'success');
        closeCreateRetentionModal();
    } catch (error) {
        console.error('Error creating retention policy:', error);
        showNotification('Failed to create retention policy: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Explainability Functions
async function generateExplanation() {
    const modelId = document.getElementById('explain-model-select').value;
    const method = document.getElementById('explain-method-select').value;
    
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        if (method === 'importance') {
            // Use GET endpoint for feature importance
            await getFeatureImportance(modelId);
            return;
        }
        
        const endpoint = method === 'shap' 
            ? `/monitoring/models/${modelId}/explain/shap`
            : `/monitoring/models/${modelId}/explain/lime`;
        
        // Note: These endpoints might need sample data
        const result = await apiCall(endpoint, {
            method: 'POST',
            body: JSON.stringify({ sample_data: {} })
        });
        
        const container = document.getElementById('explainability-display');
        container.innerHTML = `
            <div class="space-y-4">
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-medium text-blue-800 mb-2">${method.toUpperCase()} Explanation</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error generating explanation: ' + error.message, 'error');
    }
}

// Data Quality Functions
async function loadDataQualityForModel() {
    const modelId = document.getElementById('data-quality-model-select').value;
    if (!modelId) {
        document.getElementById('data-quality-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to check data quality</div>';
        return;
    }
}

async function runDataQualityCheck() {
    const modelId = document.getElementById('data-quality-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/data-quality/metrics`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('data-quality-display');
        container.innerHTML = `
            <div class="space-y-4">
                <div class="grid grid-cols-3 gap-4">
                    <div class="bg-blue-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Completeness</div>
                        <div class="text-2xl font-bold text-blue-600">${((result.completeness || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Validity</div>
                        <div class="text-2xl font-bold text-green-600">${((result.validity || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="bg-purple-50 p-4 rounded">
                        <div class="text-sm text-gray-600">Consistency</div>
                        <div class="text-2xl font-bold text-purple-600">${((result.consistency || 0) * 100).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 class="font-medium text-gray-800 mb-2">Full Metrics</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error checking data quality: ' + error.message, 'error');
    }
}

async function detectOutliers() {
    const modelId = document.getElementById('data-quality-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/data-quality/outliers`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('data-quality-display');
        const existingHTML = container.innerHTML;
        container.innerHTML = existingHTML + `
            <div class="mt-4 bg-orange-50 border border-orange-200 rounded-lg p-4">
                <h4 class="font-medium text-orange-800 mb-2">Outlier Detection</h4>
                <div class="text-sm text-orange-700">
                    <p><strong>Outliers Detected:</strong> ${result.outliers_count || 0}</p>
                    <p><strong>Outlier Percentage:</strong> ${((result.outlier_percentage || 0) * 100).toFixed(2)}%</p>
                </div>
                ${result.outliers && result.outliers.length > 0 ? `
                    <div class="mt-2 max-h-48 overflow-y-auto">
                        <p class="text-xs font-medium mb-1">Outlier Details:</p>
                        <pre class="text-xs bg-white p-2 rounded">${JSON.stringify(result.outliers.slice(0, 10), null, 2)}${result.outliers.length > 10 ? '\n... (showing first 10)' : ''}</pre>
                    </div>
                ` : ''}
                <div class="mt-2 bg-white p-2 rounded">
                    <pre class="text-xs overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error detecting outliers: ' + error.message, 'error');
    }
}

async function detectAnomaly() {
    const modelId = document.getElementById('data-quality-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/data-quality/anomaly`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('data-quality-display');
        const existingHTML = container.innerHTML;
        container.innerHTML = existingHTML + `
            <div class="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 class="font-medium text-red-800 mb-2">Anomaly Detection</h4>
                <div class="text-sm text-red-700">
                    <p><strong>Anomalies Detected:</strong> ${result.anomalies_count || 0}</p>
                    <p><strong>Anomaly Score:</strong> ${(result.anomaly_score || 0).toFixed(3)}</p>
                </div>
                ${result.anomalies && result.anomalies.length > 0 ? `
                    <div class="mt-2 max-h-48 overflow-y-auto">
                        <p class="text-xs font-medium mb-1">Anomaly Details:</p>
                        <pre class="text-xs bg-white p-2 rounded">${JSON.stringify(result.anomalies.slice(0, 10), null, 2)}${result.anomalies.length > 10 ? '\n... (showing first 10)' : ''}</pre>
                    </div>
                ` : ''}
                <div class="mt-2 bg-white p-2 rounded">
                    <pre class="text-xs overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error detecting anomalies: ' + error.message, 'error');
    }
}

// Fairness Functions
async function loadFairnessMetrics() {
    const modelId = document.getElementById('fairness-model-select').value;
    if (!modelId) {
        document.getElementById('fairness-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to calculate fairness metrics</div>';
        return;
    }
}

async function calculateFairnessMetrics() {
    const modelId = document.getElementById('fairness-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/fairness/metrics`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('fairness-display');
        container.innerHTML = `
            <div class="space-y-4">
                <div class="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                    <h4 class="font-medium text-indigo-800 mb-2">Fairness Metrics</h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error calculating fairness metrics: ' + error.message, 'error');
    }
}

async function configureFairnessAttributes() {
    const modelId = document.getElementById('fairness-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    const modal = document.getElementById('fairness-attributes-modal');
    if (modal) {
        document.getElementById('fairness-attributes-model-id').value = modelId;
        modal.classList.remove('hidden');
    } else {
        showNotification('Fairness attributes modal not found', 'error');
    }
}

function closeFairnessAttributesModal() {
    document.getElementById('fairness-attributes-modal').classList.add('hidden');
}

async function submitFairnessAttributes(e) {
    e.preventDefault();
    const modelId = document.getElementById('fairness-attributes-model-id').value;
    const protectedAttributes = document.getElementById('fairness-protected-attributes').value.split(',').map(a => a.trim()).filter(a => a);
    const sensitiveGroups = document.getElementById('fairness-sensitive-groups').value.split(',').map(g => g.trim()).filter(g => g);
    
    try {
        await apiCall(`/monitoring/models/${modelId}/fairness/attributes`, {
            method: 'POST',
            body: JSON.stringify({
                protected_attributes: protectedAttributes,
                sensitive_groups: sensitiveGroups
            })
        });
        
        showNotification('Fairness attributes configured successfully', 'success');
        closeFairnessAttributesModal();
    } catch (error) {
        console.error('Error configuring fairness attributes:', error);
        showNotification('Failed to configure fairness attributes: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function getDemographicDistribution() {
    const modelId = document.getElementById('fairness-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const demographics = await apiCall(`/monitoring/models/${modelId}/fairness/demographics`);
        
        const container = document.getElementById('fairness-display');
        const existingHTML = container.innerHTML;
        container.innerHTML = existingHTML + `
            <div class="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 class="font-medium text-blue-800 mb-2">Demographic Distribution</h4>
                <div class="space-y-2">
                    ${Object.entries(demographics.distribution || demographics).map(([group, count]) => `
                        <div class="flex items-center justify-between bg-white p-2 rounded">
                            <span class="text-sm">${group}</span>
                            <span class="text-sm font-mono">${count}</span>
                        </div>
                    `).join('')}
                </div>
                <div class="mt-2 bg-white p-2 rounded">
                    <pre class="text-xs overflow-x-auto">${JSON.stringify(demographics, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error getting demographic distribution:', error);
        showNotification('Failed to get demographic distribution: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Degradation Functions
async function loadDegradationForModel() {
    const modelId = document.getElementById('degradation-model-select').value;
    if (!modelId) {
        document.getElementById('degradation-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to detect performance degradation</div>';
        return;
    }
}

async function detectDegradation() {
    const modelId = document.getElementById('degradation-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/degradation/detect`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('degradation-display');
        container.innerHTML = `
            <div class="space-y-4">
                <div class="bg-${result.degradation_detected ? 'red' : 'green'}-50 border border-${result.degradation_detected ? 'red' : 'green'}-200 rounded-lg p-4">
                    <h4 class="font-medium text-${result.degradation_detected ? 'red' : 'green'}-800 mb-2">
                        ${result.degradation_detected ? 'Degradation Detected' : 'No Degradation Detected'}
                    </h4>
                    <pre class="text-xs bg-white p-3 rounded overflow-x-auto mt-2">${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        showNotification('Error detecting degradation: ' + error.message, 'error');
    }
}

// Populate model selectors when monitoring tab is shown
function populateMonitoringModelSelectors() {
    loadModels().then(models => {
        const selectors = [
            'performance-model-select',
            'drift-model-select',
            'logs-model-select',
            'explain-model-select',
            'data-quality-model-select',
            'fairness-model-select',
            'degradation-model-select'
        ];
        
        selectors.forEach(selectorId => {
            const select = document.getElementById(selectorId);
            if (select) {
                const currentValue = select.value;
                select.innerHTML = '<option value="">Select a model...</option>';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = `${model.name} (${model.framework})`;
                    select.appendChild(option);
                });
                if (currentValue) {
                    select.value = currentValue;
                }
            }
        });
    });
}

// showTab function is already defined above with all tab handlers - no override needed

// Export monitoring functions
window.loadMonitoringData = loadMonitoringData;
window.loadSystemHealth = loadSystemHealth;
window.loadAlerts = loadAlerts;
window.resolveAlert = resolveAlert;
window.detectDrift = detectDrift;
window.loadModelPerformance = loadModelPerformance;
window.loadPredictionLogs = loadPredictionLogs;
window.loadAggregatedMetrics = loadAggregatedMetrics;
window.loadDeploymentSummary = loadDeploymentSummary;
window.loadConfidenceMetrics = loadConfidenceMetrics;
window.loadResourceUsage = loadResourceUsage;
window.loadModelPerformanceForUI = loadModelPerformanceForUI;
window.loadDriftForModel = loadDriftForModel;
window.runDriftDetection = runDriftDetection;
window.loadPredictionLogsForUI = loadPredictionLogsForUI;
window.viewModelDetails = viewModelDetails;
window.closeModelDetailsModal = closeModelDetailsModal;
window.viewDeploymentLogs = viewDeploymentLogs;
window.closeDeploymentLogsModal = closeDeploymentLogsModal;
window.showSchemaTab = showSchemaTab;
window.openCreateABTestModal = openCreateABTestModal;
window.closeCreateABTestModal = closeCreateABTestModal;
window.openCreateCanaryModal = openCreateCanaryModal;
window.closeCreateCanaryModal = closeCreateCanaryModal;
window.showGovernanceTab = showGovernanceTab;
window.generateExplanation = generateExplanation;
window.loadDataQualityForModel = loadDataQualityForModel;
window.runDataQualityCheck = runDataQualityCheck;
window.loadFairnessMetrics = loadFairnessMetrics;
window.calculateFairnessMetrics = calculateFairnessMetrics;
window.loadDegradationForModel = loadDegradationForModel;
window.detectDegradation = detectDegradation;

// Alert Rules Functions
function openAlertRulesModal() {
    document.getElementById('alert-rules-modal').classList.remove('hidden');
    loadAlertRules();
}

function closeAlertRulesModal() {
    document.getElementById('alert-rules-modal').classList.add('hidden');
}

async function loadAlertRules() {
    try {
        // Note: Backend might need a GET endpoint for listing rules
        const container = document.getElementById('alert-rules-list');
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>Alert rule listing requires backend GET endpoint</p>
                <p class="text-sm mt-2">Use the "Create Rule" button to create a new alert rule</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading alert rules:', error);
    }
}

// Alert Management Functions
function openCreateAlertRuleModal() {
    const modal = document.getElementById('create-alert-rule-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Create alert rule modal not found', 'error');
    }
}

function closeCreateAlertRuleModal() {
    document.getElementById('create-alert-rule-modal').classList.add('hidden');
}

async function submitCreateAlertRule(e) {
    e.preventDefault();
    const ruleData = {
        name: document.getElementById('alert-rule-name').value,
        description: document.getElementById('alert-rule-description').value,
        metric: document.getElementById('alert-rule-metric').value,
        threshold: parseFloat(document.getElementById('alert-rule-threshold').value),
        operator: document.getElementById('alert-rule-operator').value,
        severity: document.getElementById('alert-rule-severity').value,
        model_id: document.getElementById('alert-rule-model-id').value || null,
        enabled: document.getElementById('alert-rule-enabled').checked
    };
    
    try {
        await apiCall('/monitoring/alert-rules', {
            method: 'POST',
            body: JSON.stringify(ruleData)
        });
        
        showNotification('Alert rule created successfully', 'success');
        closeCreateAlertRuleModal();
        loadAlertRules();
    } catch (error) {
        console.error('Error creating alert rule:', error);
        showNotification('Failed to create alert rule: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function loadAlertRules() {
    try {
        // Note: Backend might need a GET endpoint for listing alert rules
        const container = document.getElementById('alert-rules-list');
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>Alert rules listing requires backend GET endpoint</p>
                <p class="text-sm mt-2">Use the "Create Rule" button to create a new alert rule</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading alert rules:', error);
    }
}

async function acknowledgeAlert(alertId) {
    try {
        await apiCall(`/monitoring/alerts/${alertId}/acknowledge`, {
            method: 'POST'
        });
        showNotification('Alert acknowledged successfully', 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error acknowledging alert:', error);
        showNotification('Failed to acknowledge alert: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function checkAndCreateAlerts() {
    try {
        const alerts = await apiCall('/monitoring/alerts/check', {
            method: 'POST'
        });
        showNotification(`Checked alerts: ${alerts.length} new alerts created`, 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error checking alerts:', error);
        showNotification('Failed to check alerts: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateNotificationChannelModal() {
    const modal = document.getElementById('create-notification-channel-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Notification channel modal not found', 'error');
    }
}

function closeCreateNotificationChannelModal() {
    document.getElementById('create-notification-channel-modal').classList.add('hidden');
}

async function submitCreateNotificationChannel(e) {
    e.preventDefault();
    const channelData = {
        name: document.getElementById('notification-channel-name').value,
        type: document.getElementById('notification-channel-type').value,
        config: JSON.parse(document.getElementById('notification-channel-config').value)
    };
    
    try {
        await apiCall('/monitoring/notifications/channels', {
            method: 'POST',
            body: JSON.stringify(channelData)
        });
        
        showNotification('Notification channel created successfully', 'success');
        closeCreateNotificationChannelModal();
    } catch (error) {
        console.error('Error creating notification channel:', error);
        showNotification('Failed to create notification channel: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function sendAlertNotification() {
    const alertId = document.getElementById('send-notification-alert-id').value;
    const channelId = document.getElementById('send-notification-channel-id').value;
    
    try {
        await apiCall('/monitoring/notifications/send', {
            method: 'POST',
            body: JSON.stringify({
                alert_id: alertId,
                channel_id: channelId
            })
        });
        
        showNotification('Notification sent successfully', 'success');
    } catch (error) {
        console.error('Error sending notification:', error);
        showNotification('Failed to send notification: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function groupAlerts() {
    const alertIds = document.getElementById('group-alert-ids').value.split(',').map(id => id.trim());
    
    try {
        await apiCall('/monitoring/alerts/group', {
            method: 'POST',
            body: JSON.stringify({ alert_ids: alertIds })
        });
        
        showNotification('Alerts grouped successfully', 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error grouping alerts:', error);
        showNotification('Failed to group alerts: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateAlertEscalationModal() {
    const modal = document.getElementById('create-alert-escalation-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Alert escalation modal not found', 'error');
    }
}

function closeCreateAlertEscalationModal() {
    document.getElementById('create-alert-escalation-modal').classList.add('hidden');
}

async function submitCreateAlertEscalation(e) {
    e.preventDefault();
    const escalationData = {
        name: document.getElementById('escalation-name').value,
        rules: JSON.parse(document.getElementById('escalation-rules').value),
        actions: JSON.parse(document.getElementById('escalation-actions').value)
    };
    
    try {
        await apiCall('/monitoring/alerts/escalations', {
            method: 'POST',
            body: JSON.stringify(escalationData)
        });
        
        showNotification('Alert escalation created successfully', 'success');
        closeCreateAlertEscalationModal();
    } catch (error) {
        console.error('Error creating alert escalation:', error);
        showNotification('Failed to create escalation: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function checkAndEscalateAlerts() {
    try {
        const escalated = await apiCall('/monitoring/alerts/escalate', {
            method: 'POST'
        });
        showNotification(`Escalated ${escalated.length || 0} alerts`, 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error escalating alerts:', error);
        showNotification('Failed to escalate alerts: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function createAlert() {
    const alertData = {
        model_id: document.getElementById('create-alert-model-id').value || null,
        metric: document.getElementById('create-alert-metric').value,
        value: parseFloat(document.getElementById('create-alert-value').value),
        severity: document.getElementById('create-alert-severity').value,
        message: document.getElementById('create-alert-message').value
    };
    
    try {
        await apiCall('/monitoring/alerts', {
            method: 'POST',
            body: JSON.stringify(alertData)
        });
        
        showNotification('Alert created successfully', 'success');
        loadAlerts();
    } catch (error) {
        console.error('Error creating alert:', error);
        showNotification('Failed to create alert: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Baseline Management Functions
async function loadBaselineForModel() {
    const modelId = document.getElementById('baseline-model-select').value;
    if (!modelId) {
        document.getElementById('baseline-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to manage baseline</div>';
        return;
    }
    
    try {
        const baseline = await apiCall(`/monitoring/models/${modelId}/baseline`);
        renderBaseline(baseline);
    } catch (error) {
        document.getElementById('baseline-display').innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <p>No baseline found for this model</p>
                <button onclick="createBaseline()" class="mt-2 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                    Create Baseline
                </button>
            </div>
        `;
    }
}

async function createBaseline() {
    const modelId = document.getElementById('baseline-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/baseline`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        showNotification('Baseline created successfully', 'success');
        loadBaselineForModel();
    } catch (error) {
        showNotification('Error creating baseline: ' + error.message, 'error');
    }
}

function renderBaseline(baseline) {
    const container = document.getElementById('baseline-display');
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 class="font-medium text-blue-800 mb-2">Baseline Information</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(baseline, null, 2)}</pre>
            </div>
            <button onclick="compareWithBaseline()" class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded">
                Compare Current Performance with Baseline
            </button>
        </div>
    `;
}

async function compareWithBaseline() {
    const modelId = document.getElementById('baseline-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const result = await apiCall(`/monitoring/models/${modelId}/baseline/compare`, {
            method: 'POST',
            body: JSON.stringify({})
        });
        
        const container = document.getElementById('baseline-display');
        container.innerHTML += `
            <div class="mt-4 bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 class="font-medium text-gray-800 mb-2">Comparison Results</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
    } catch (error) {
        showNotification('Error comparing with baseline: ' + error.message, 'error');
    }
}

// Analytics Functions
function showAnalyticsTab(tabName) {
    document.querySelectorAll('.analytics-tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelectorAll('.analytics-tab-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'text-blue-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    
    document.getElementById(`analytics-${tabName}`).classList.remove('hidden');
    event.target.classList.remove('border-transparent', 'text-gray-500');
    event.target.classList.add('border-blue-500', 'text-blue-600');
}

document.addEventListener('DOMContentLoaded', function() {
    const timeSeriesForm = document.getElementById('time-series-form');
    if (timeSeriesForm) {
        timeSeriesForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const metricName = document.getElementById('time-series-metric').value;
                const modelId = document.getElementById('time-series-model').value;
                
                const params = new URLSearchParams({ metric_name: metricName });
                if (modelId) params.append('model_id', modelId);
                
                const result = await apiCall(`/monitoring/analytics/time-series?${params}`, {
                    method: 'POST'
                });
                
                const resultDiv = document.getElementById('time-series-result');
                resultDiv.innerHTML = `
                    <div class="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <h4 class="font-medium text-blue-800 mb-2">Time Series Analysis</h4>
                        <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
                    </div>
                `;
            } catch (error) {
                showNotification('Error analyzing time series: ' + error.message, 'error');
            }
        });
    }
});

function openCreateComparativeModal() {
    showNotification('Comparative analytics creation form coming soon', 'info');
}

// Analytics Functions
async function createComparativeAnalytics(e) {
    e.preventDefault();
    const analyticsData = {
        name: document.getElementById('comparative-analytics-name').value,
        description: document.getElementById('comparative-analytics-description').value,
        models: document.getElementById('comparative-analytics-models').value.split(',').map(m => m.trim()),
        metrics: document.getElementById('comparative-analytics-metrics').value.split(',').map(m => m.trim()),
        time_range: document.getElementById('comparative-analytics-time-range').value
    };
    
    try {
        const result = await apiCall('/monitoring/analytics/comparative', {
            method: 'POST',
            body: JSON.stringify(analyticsData)
        });
        
        showNotification('Comparative analytics created successfully', 'success');
        document.getElementById('comparative-list').innerHTML = `
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 class="font-medium text-green-800 mb-2">Comparison Created</h4>
                <pre class="text-xs bg-white p-3 rounded overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
    } catch (error) {
        console.error('Error creating comparative analytics:', error);
        showNotification('Failed to create comparative analytics: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateDashboardModal() {
    const modal = document.getElementById('create-dashboard-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Dashboard modal not found', 'error');
    }
}

function closeCreateDashboardModal() {
    document.getElementById('create-dashboard-modal').classList.add('hidden');
}

async function submitCreateDashboard(e) {
    e.preventDefault();
    const dashboardData = {
        name: document.getElementById('dashboard-name').value,
        description: document.getElementById('dashboard-description').value,
        widgets: JSON.parse(document.getElementById('dashboard-widgets').value || '[]'),
        layout: document.getElementById('dashboard-layout').value,
        refresh_interval: parseInt(document.getElementById('dashboard-refresh-interval').value || '60')
    };
    
    try {
        await apiCall('/monitoring/analytics/dashboards', {
            method: 'POST',
            body: JSON.stringify(dashboardData)
        });
        
        showNotification('Analytics dashboard created successfully', 'success');
        closeCreateDashboardModal();
    } catch (error) {
        console.error('Error creating dashboard:', error);
        showNotification('Failed to create dashboard: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateReportModal() {
    const modal = document.getElementById('create-report-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Report modal not found', 'error');
    }
}

function closeCreateReportModal() {
    document.getElementById('create-report-modal').classList.add('hidden');
}

async function submitCreateReport(e) {
    e.preventDefault();
    const reportData = {
        name: document.getElementById('report-name').value,
        description: document.getElementById('report-description').value,
        report_type: document.getElementById('report-type').value,
        schedule: document.getElementById('report-schedule').value,
        recipients: document.getElementById('report-recipients').value.split(',').map(r => r.trim()),
        metrics: JSON.parse(document.getElementById('report-metrics').value || '[]'),
        format: document.getElementById('report-format').value
    };
    
    try {
        await apiCall('/monitoring/analytics/reports', {
            method: 'POST',
            body: JSON.stringify(reportData)
        });
        
        showNotification('Analytics report created successfully', 'success');
        closeCreateReportModal();
    } catch (error) {
        console.error('Error creating report:', error);
        showNotification('Failed to create report: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Model Lifecycle Functions
async function loadModelCard() {
    const modelId = document.getElementById('lifecycle-model-select').value;
    if (!modelId) {
        document.getElementById('model-card-display').innerHTML = 
            '<div class="text-center py-4 text-gray-500">Select a model to view or generate model card</div>';
        return;
    }
    
    try {
        const card = await apiCall(`/monitoring/models/${modelId}/card`);
        renderModelCard(card);
    } catch (error) {
        document.getElementById('model-card-display').innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <p>No model card found</p>
                <button onclick="generateModelCard()" class="mt-2 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                    Generate Model Card
                </button>
            </div>
        `;
    }
}

async function generateModelCard() {
    const modelId = document.getElementById('lifecycle-model-select').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    try {
        const card = await apiCall(`/monitoring/models/${modelId}/card/generate`, {
            method: 'POST'
        });
        showNotification('Model card generated successfully', 'success');
        renderModelCard(card);
    } catch (error) {
        showNotification('Error generating model card: ' + error.message, 'error');
    }
}

function renderModelCard(card) {
    const container = document.getElementById('model-card-display');
    container.innerHTML = `
        <div class="space-y-4">
            <div class="bg-white border border-gray-200 rounded-lg p-6">
                <h4 class="text-lg font-medium text-gray-900 mb-4">Model Card</h4>
                <div class="prose max-w-none">
                    <pre class="text-xs bg-gray-50 p-4 rounded overflow-x-auto">${JSON.stringify(card, null, 2)}</pre>
                </div>
            </div>
        </div>
    `;
}

// Lifecycle Functions
function openCreateRetrainingModal() {
    const modal = document.getElementById('create-retraining-modal');
    if (modal) {
        populateMonitoringModelSelectors();
        modal.classList.remove('hidden');
    } else {
        showNotification('Retraining modal not found', 'error');
    }
}

function closeCreateRetrainingModal() {
    document.getElementById('create-retraining-modal').classList.add('hidden');
}

async function submitCreateRetrainingJob(e) {
    e.preventDefault();
    const jobData = {
        model_id: document.getElementById('retraining-model-id').value,
        training_data_path: document.getElementById('retraining-data-path').value,
        hyperparameters: JSON.parse(document.getElementById('retraining-hyperparameters').value || '{}'),
        description: document.getElementById('retraining-description').value
    };
    
    try {
        await apiCall(`/monitoring/models/${jobData.model_id}/retraining/jobs`, {
            method: 'POST',
            body: JSON.stringify(jobData)
        });
        
        showNotification('Retraining job created successfully', 'success');
        closeCreateRetrainingModal();
        loadRetrainingJobs();
    } catch (error) {
        console.error('Error creating retraining job:', error);
        showNotification('Failed to create retraining job: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function configureRetrainingTrigger() {
    const modelId = document.getElementById('retraining-model-id').value;
    if (!modelId) {
        showNotification('Please select a model', 'error');
        return;
    }
    
    const triggerData = {
        trigger_type: document.getElementById('retraining-trigger-type').value,
        threshold: parseFloat(document.getElementById('retraining-trigger-threshold').value),
        schedule: document.getElementById('retraining-trigger-schedule').value,
        enabled: document.getElementById('retraining-trigger-enabled').checked
    };
    
    try {
        await apiCall(`/monitoring/models/${modelId}/retraining/triggers`, {
            method: 'POST',
            body: JSON.stringify(triggerData)
        });
        
        showNotification('Retraining trigger configured successfully', 'success');
    } catch (error) {
        console.error('Error configuring retraining trigger:', error);
        showNotification('Failed to configure trigger: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Integration Functions
function openCreateIntegrationModal() {
    const modal = document.getElementById('create-integration-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Integration modal not found', 'error');
    }
}

function closeCreateIntegrationModal() {
    document.getElementById('create-integration-modal').classList.add('hidden');
}

async function submitCreateIntegration(e) {
    e.preventDefault();
    const integrationData = {
        name: document.getElementById('integration-name').value,
        type: document.getElementById('integration-type').value,
        config: JSON.parse(document.getElementById('integration-config').value || '{}'),
        enabled: document.getElementById('integration-enabled').checked
    };
    
    try {
        await apiCall('/monitoring/integrations', {
            method: 'POST',
            body: JSON.stringify(integrationData)
        });
        
        showNotification('Integration created successfully', 'success');
        closeCreateIntegrationModal();
        loadIntegrations();
    } catch (error) {
        console.error('Error creating integration:', error);
        showNotification('Failed to create integration: ' + (error.message || 'Unknown error'), 'error');
    }
}

function openCreateWebhookModal() {
    const modal = document.getElementById('create-webhook-modal');
    if (modal) {
        modal.classList.remove('hidden');
    } else {
        showNotification('Webhook modal not found', 'error');
    }
}

function closeCreateWebhookModal() {
    document.getElementById('create-webhook-modal').classList.add('hidden');
}

async function submitCreateWebhook(e) {
    e.preventDefault();
    const webhookData = {
        url: document.getElementById('webhook-url').value,
        events: document.getElementById('webhook-events').value.split(',').map(e => e.trim()),
        secret: document.getElementById('webhook-secret').value,
        enabled: document.getElementById('webhook-enabled').checked
    };
    
    try {
        await apiCall('/monitoring/integrations/webhooks', {
            method: 'POST',
            body: JSON.stringify(webhookData)
        });
        
        showNotification('Webhook created successfully', 'success');
        closeCreateWebhookModal();
    } catch (error) {
        console.error('Error creating webhook:', error);
        showNotification('Failed to create webhook: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function createSamplingConfig() {
    const configData = {
        model_id: document.getElementById('sampling-model-id').value,
        sampling_rate: parseFloat(document.getElementById('sampling-rate').value),
        sampling_strategy: document.getElementById('sampling-strategy').value,
        enabled: document.getElementById('sampling-enabled').checked
    };
    
    try {
        await apiCall('/monitoring/integrations/sampling', {
            method: 'POST',
            body: JSON.stringify(configData)
        });
        
        showNotification('Sampling configuration created successfully', 'success');
    } catch (error) {
        console.error('Error creating sampling config:', error);
        showNotification('Failed to create sampling config: ' + (error.message || 'Unknown error'), 'error');
    }
}

async function createAggregationConfig() {
    const configData = {
        metric_name: document.getElementById('aggregation-metric-name').value,
        aggregation_type: document.getElementById('aggregation-type').value,
        window_size: parseInt(document.getElementById('aggregation-window-size').value),
        enabled: document.getElementById('aggregation-enabled').checked
    };
    
    try {
        await apiCall('/monitoring/integrations/aggregation', {
            method: 'POST',
            body: JSON.stringify(configData)
        });
        
        showNotification('Aggregation configuration created successfully', 'success');
    } catch (error) {
        console.error('Error creating aggregation config:', error);
        showNotification('Failed to create aggregation config: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Audit Logs Functions
async function loadAuditLogs() {
    try {
        const entityType = document.getElementById('audit-entity-filter').value;
        const params = entityType ? `?entity_type=${entityType}` : '';
        const logs = await apiCall(`/monitoring/audit${params}`);
        renderAuditLogs(logs);
    } catch (error) {
        console.error('Error loading audit logs:', error);
        document.getElementById('audit-logs-display').innerHTML = 
            '<p class="text-gray-500 text-center py-4">Error loading audit logs</p>';
    }
}

function renderAuditLogs(logs) {
    const container = document.getElementById('audit-logs-display');
    if (!logs || logs.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No audit logs found</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="space-y-2 max-h-96 overflow-y-auto">
            ${logs.map(log => `
                <div class="border rounded p-3 bg-gray-50">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <p class="text-sm font-medium">${log.action || 'Unknown Action'}</p>
                            <p class="text-xs text-gray-600 mt-1">Entity: ${log.entity_type || 'N/A'} - ${log.entity_id || 'N/A'}</p>
                            <p class="text-xs text-gray-500 mt-1">${new Date(log.timestamp || log.created_at).toLocaleString()}</p>
                            ${log.user_id ? `<p class="text-xs text-gray-500">User: ${log.user_id}</p>` : ''}
                        </div>
                        <span class="px-2 py-1 rounded text-xs bg-blue-100 text-blue-800">
                            ${log.status || 'success'}
                        </span>
                    </div>
                    ${log.details ? `
                        <div class="mt-2">
                            <pre class="text-xs bg-white p-2 rounded overflow-x-auto">${JSON.stringify(log.details, null, 2)}</pre>
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

// Update populateMonitoringModelSelectors to include baseline selector
function populateMonitoringModelSelectors() {
    loadModels().then(models => {
        const selectors = [
            'performance-model-select',
            'drift-model-select',
            'logs-model-select',
            'explain-model-select',
            'data-quality-model-select',
            'fairness-model-select',
            'degradation-model-select',
            'baseline-model-select',
            'lifecycle-model-select',
            'time-series-model'
        ];
        
        selectors.forEach(selectorId => {
            const select = document.getElementById(selectorId);
            if (select) {
                const currentValue = select.value;
                select.innerHTML = '<option value="">Select a model...</option>';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = `${model.name} (${model.framework})`;
                    select.appendChild(option);
                });
                if (currentValue) {
                    select.value = currentValue;
                }
            }
        });
    });
}

// Export new functions
window.openAlertRulesModal = openAlertRulesModal;
window.closeAlertRulesModal = closeAlertRulesModal;
window.loadBaselineForModel = loadBaselineForModel;
window.createBaseline = createBaseline;
window.compareWithBaseline = compareWithBaseline;
window.showAnalyticsTab = showAnalyticsTab;
window.openCreateComparativeModal = openCreateComparativeModal;
window.openCreateDashboardModal = openCreateDashboardModal;
window.openCreateReportModal = openCreateReportModal;
window.loadModelCard = loadModelCard;
window.generateModelCard = generateModelCard;
window.openCreateRetrainingModal = openCreateRetrainingModal;
window.openCreateIntegrationModal = openCreateIntegrationModal;
window.openCreateWebhookModal = openCreateWebhookModal;
window.loadAuditLogs = loadAuditLogs;
window.startDeployment = startDeployment;
window.deleteDeployment = deleteDeployment;
window.getDeploymentStatus = getDeploymentStatus;
window.getDeploymentMetrics = getDeploymentMetrics;
window.testDeploymentEndpoint = testDeploymentEndpoint;
window.submitDeploymentTest = submitDeploymentTest;
window.openUpdateDeploymentModal = openUpdateDeploymentModal;
window.closeUpdateDeploymentModal = closeUpdateDeploymentModal;
window.submitUpdateDeployment = submitUpdateDeployment;
window.closeDeploymentStatusModal = closeDeploymentStatusModal;
window.closeDeploymentMetricsModal = closeDeploymentMetricsModal;
window.closeDeploymentTestModal = closeDeploymentTestModal;
window.openUpdateModelModal = openUpdateModelModal;
window.closeUpdateModelModal = closeUpdateModelModal;
window.submitUpdateModel = submitUpdateModel;
window.validateModel = validateModel;
window.getModelMetrics = getModelMetrics;
window.closeModelMetricsModal = closeModelMetricsModal;
window.getABTestDetails = getABTestDetails;
window.closeABTestDetailsModal = closeABTestDetailsModal;
window.startABTest = startABTest;
window.stopABTest = stopABTest;
window.getABTestMetrics = getABTestMetrics;
window.assignABTestVariant = assignABTestVariant;
window.getCanaryDetails = getCanaryDetails;
window.closeCanaryDetailsModal = closeCanaryDetailsModal;
window.startCanaryRollout = startCanaryRollout;
window.advanceCanaryRollout = advanceCanaryRollout;
window.rollbackCanary = rollbackCanary;
window.getCanaryMetrics = getCanaryMetrics;
window.checkCanaryHealth = checkCanaryHealth;
window.closeCanaryHealthModal = closeCanaryHealthModal;
window.logPredictionWithGroundTruth = logPredictionWithGroundTruth;
window.closeDegradationLogModal = closeDegradationLogModal;
window.submitDegradationLog = submitDegradationLog;
window.compareModelVersions = compareModelVersions;
window.closeVersionComparisonModal = closeVersionComparisonModal;
window.submitVersionComparison = submitVersionComparison;
window.populateSchemaModelSelector = populateSchemaModelSelector;
window.loadModelSchemas = loadModelSchemas;
window.getSchemaExample = getSchemaExample;
window.getOpenAPISchema = getOpenAPISchema;
window.closeOpenAPISchemaModal = closeOpenAPISchemaModal;
window.validateModelSchema = validateModelSchema;
window.openUpdateModelSchemaModal = openUpdateModelSchemaModal;
window.closeUpdateModelSchemaModal = closeUpdateModelSchemaModal;
window.submitUpdateModelSchema = submitUpdateModelSchema;
window.deleteModelSchema = deleteModelSchema;
window.loadSchemaVersions = loadSchemaVersions;
window.openCreateSchemaVersionModal = openCreateSchemaVersionModal;
window.closeCreateSchemaVersionModal = closeCreateSchemaVersionModal;
window.submitCreateSchemaVersion = submitCreateSchemaVersion;
window.viewSchemaVersion = viewSchemaVersion;
window.closeSchemaVersionViewModal = closeSchemaVersionViewModal;
window.updateSchema = updateSchema;
window.closeUpdateSchemaModal = closeUpdateSchemaModal;
window.submitUpdateSchema = submitUpdateSchema;
window.deleteSchema = deleteSchema;
window.loadCommonTemplates = loadCommonTemplates;
window.closeCreateAlertRuleModal = closeCreateAlertRuleModal;
window.submitCreateAlertRule = submitCreateAlertRule;
window.loadAlertRules = loadAlertRules;
window.acknowledgeAlert = acknowledgeAlert;
window.checkAndCreateAlerts = checkAndCreateAlerts;
window.openCreateNotificationChannelModal = openCreateNotificationChannelModal;
window.closeCreateNotificationChannelModal = closeCreateNotificationChannelModal;
window.submitCreateNotificationChannel = submitCreateNotificationChannel;
window.sendAlertNotification = sendAlertNotification;
window.groupAlerts = groupAlerts;
window.openCreateAlertEscalationModal = openCreateAlertEscalationModal;
window.closeCreateAlertEscalationModal = closeCreateAlertEscalationModal;
window.submitCreateAlertEscalation = submitCreateAlertEscalation;
window.checkAndEscalateAlerts = checkAndEscalateAlerts;
window.createAlert = createAlert;
window.closeCreateLineageModal = closeCreateLineageModal;
window.submitCreateLineage = submitCreateLineage;
window.closeCreateWorkflowModal = closeCreateWorkflowModal;
window.submitCreateWorkflow = submitCreateWorkflow;
window.closeCreateComplianceModal = closeCreateComplianceModal;
window.submitCreateCompliance = submitCreateCompliance;
window.closeCreateRetentionModal = closeCreateRetentionModal;
window.submitCreateRetentionPolicy = submitCreateRetentionPolicy;
window.createComparativeAnalytics = createComparativeAnalytics;
window.closeCreateDashboardModal = closeCreateDashboardModal;
window.submitCreateDashboard = submitCreateDashboard;
window.closeCreateReportModal = closeCreateReportModal;
window.submitCreateReport = submitCreateReport;
window.closeCreateRetrainingModal = closeCreateRetrainingModal;
window.submitCreateRetrainingJob = submitCreateRetrainingJob;
window.configureRetrainingTrigger = configureRetrainingTrigger;
window.closeCreateIntegrationModal = closeCreateIntegrationModal;
window.submitCreateIntegration = submitCreateIntegration;
window.closeCreateWebhookModal = closeCreateWebhookModal;
window.submitCreateWebhook = submitCreateWebhook;
window.createSamplingConfig = createSamplingConfig;
window.createAggregationConfig = createAggregationConfig;
window.showSchemaTab = showSchemaTab; 