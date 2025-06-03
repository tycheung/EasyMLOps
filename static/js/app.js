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
        const models = await apiCall('/models');
        currentModels = models;
        document.getElementById('total-models').textContent = models.length;
        
        // Load deployments
        const deployments = await apiCall('/deployments');
        currentDeployments = deployments;
        const activeDeployments = deployments.filter(d => d.status === 'active').length;
        document.getElementById('active-deployments').textContent = activeDeployments;
        
        // Mock metrics (would come from monitoring service)
        document.getElementById('predictions-today').textContent = Math.floor(Math.random() * 1000);
        document.getElementById('avg-response-time').textContent = `${Math.floor(Math.random() * 50 + 10)}ms`;
        
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
        const templates = await apiCall('/models/templates/common');
        const template = templates.templates[templateName];
        
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
        
        showNotification(`Loaded ${templateName} template`, 'success');
        
    } catch (error) {
        console.error('Error loading template:', error);
    }
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
        const models = await apiCall('/models');
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
        const deployment = await apiCall(`/deployments`, {
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
        const deployments = await apiCall('/deployments');
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
        const deployments = await apiCall('/deployments');
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
window.clearFile = clearFile;
window.loadTestInterface = loadTestInterface;
window.loadExampleData = loadExampleData;
window.viewModelDetails = viewModelDetails;
window.deployModel = deployModel;
window.deleteModel = deleteModel;
window.testDeployment = testDeployment;
window.viewDeploymentLogs = viewDeploymentLogs;
window.stopDeployment = stopDeployment; 