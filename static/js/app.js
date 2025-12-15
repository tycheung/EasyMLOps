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
                    <button onclick="viewModelDetails('${model.id}')" class="text-blue-600 hover:text-blue-800">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="deployModel('${model.id}')" class="text-green-600 hover:text-green-800">
                        <i class="fas fa-rocket"></i>
                    </button>
                    <button onclick="deleteModel('${model.id}')" class="text-red-600 hover:text-red-800">
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
    }
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
                    <h4 class="text-lg font-medium text-gray-900">${deployment.service_name}</h4>
                    <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                        <span><i class="fas fa-server mr-1"></i>${deployment.framework}</span>
                        <span><i class="fas fa-calendar mr-1"></i>${new Date(deployment.created_at).toLocaleDateString()}</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getDeploymentStatusClass(deployment.status)}">${deployment.status}</span>
                    </div>
                    ${deployment.endpoint_url ? `
                        <div class="mt-2">
                            <span class="text-xs text-gray-500">Endpoint:</span>
                            <code class="ml-1 text-xs bg-gray-100 px-2 py-1 rounded">${deployment.endpoint_url}</code>
                        </div>
                    ` : ''}
                </div>
                <div class="flex items-center space-x-2">
                    <button onclick="testDeployment('${deployment.id}')" class="text-blue-600 hover:text-blue-800">
                        <i class="fas fa-play"></i>
                    </button>
                    <button onclick="viewDeploymentLogs('${deployment.id}')" class="text-gray-600 hover:text-gray-800">
                        <i class="fas fa-file-alt"></i>
                    </button>
                    <button onclick="stopDeployment('${deployment.id}')" class="text-red-600 hover:text-red-800">
                        <i class="fas fa-stop"></i>
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
        document.getElementById('test-interface').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error loading test interface:', error);
        showNotification('Failed to load test interface', 'error');
    }
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
    }
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
    
    document.getElementById(`schema-${tabName}`).classList.remove('hidden');
    event.target.classList.remove('border-transparent', 'text-gray-500');
    event.target.classList.add('border-blue-500', 'text-blue-600');
}

// Schema Validation
document.addEventListener('DOMContentLoaded', function() {
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
});

// A/B Testing Functions
async function loadABTests() {
    try {
        // Note: Backend might need a GET endpoint for listing tests
        // For now, we'll show a message
        const container = document.getElementById('ab-tests-list');
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>A/B test listing requires backend GET endpoint</p>
                <p class="text-sm mt-2">Use the "Create A/B Test" button to create a new test</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading A/B tests:', error);
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
                loadABTests();
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
                loadCanaryDeployments();
            } catch (error) {
                showNotification('Error creating canary: ' + error.message, 'error');
            }
        });
    }
});

async function loadCanaryDeployments() {
    try {
        // Note: Backend might need a GET endpoint for listing canaries
        const container = document.getElementById('canary-list');
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>Canary listing requires backend GET endpoint</p>
                <p class="text-sm mt-2">Use the "Create Canary" button to create a new canary deployment</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading canary deployments:', error);
    }
}

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

function openCreateLineageModal() {
    showNotification('Lineage creation form coming soon', 'info');
}

function openCreateWorkflowModal() {
    showNotification('Workflow creation form coming soon', 'info');
}

function openCreateComplianceModal() {
    showNotification('Compliance record creation form coming soon', 'info');
}

function openCreateRetentionModal() {
    showNotification('Retention policy creation form coming soon', 'info');
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

function openCreateAlertRuleModal() {
    showNotification('Alert rule creation form coming soon', 'info');
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

function openCreateDashboardModal() {
    showNotification('Custom dashboard creation form coming soon', 'info');
}

function openCreateReportModal() {
    showNotification('Automated report creation form coming soon', 'info');
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

function openCreateRetrainingModal() {
    showNotification('Retraining job creation form coming soon', 'info');
}

// Integration Functions
function openCreateIntegrationModal() {
    showNotification('Integration creation form coming soon', 'info');
}

function openCreateWebhookModal() {
    showNotification('Webhook creation form coming soon', 'info');
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