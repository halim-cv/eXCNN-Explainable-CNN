/**
 * eXCNN Frontend Application
 * Handles user interactions, API calls, and UI updates
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Precomputed Examples
const EXAMPLES = [
    { id: '207_golden_retriever', name: 'Golden Retriever', icon: '' },
    { id: '281_tabby_cat', name: 'Tabby Cat', icon: '' },
    { id: '954_banana', name: 'Banana', icon: '' },
    { id: '504_coffee_mug', name: 'Coffee Mug', icon: '' },
    { id: '859_toaster', name: 'Toaster', icon: '' }
];

// Method Descriptions
// Method Descriptions
const EXPLANATIONS = {
    gradcam: {
        name: 'Grad-CAM',
        icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        desc: "Grad-CAM visualizes which regions of the image were most important for the model's prediction by projecting the gradients of the target class back to the final convolutional layer. The red and yellow areas represent high activation."
    },
    occlusion: {
        name: 'Occlusion Sensitivity',
        icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><path d="M3 9h18M9 21V9"/></svg>',
        desc: "Occlusion Sensitivity systematically hides parts of the image and measures the drop in prediction confidence. Bright spots show critical features where occlusion caused a significant confidence drop."
    },
    guidedbackprop: { // Note: key matches backend generic
        name: 'Guided Backprop',
        icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        desc: "Guided Backpropagation highlights the fine-grained pixel details that contributed positively to the prediction. It effectively visualizes the specific visual contours and patterns the network has learned."
    },
    guided_gradcam: {
        name: 'Guided Grad-CAM',
        icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        desc: "Guided Grad-CAM combines the high-resolution details of Guided Backpropagation with the class-discriminative localization of Grad-CAM. It results in the sharpest visualization."
    },
    original: {
        name: 'Original Image',
        icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
        desc: "This is the input image masked to the model's expected 224x224 dimension. It serves as the baseline reference."
    }
};

// State
let currentImage = null;
let currentFile = null;

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const analysisControls = document.getElementById('analysisControls');
const analyzeBtn = document.getElementById('analyzeBtn');
const interactiveResults = document.getElementById('interactiveResults');
const exampleDetail = document.getElementById('exampleDetail');
const backToGallery = document.getElementById('backToGallery');
const examplesGallery = document.getElementById('examplesGallery');

// History State (Session based to avoid storage limits)
let sessionHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initGallery();
    initAnalyze();
    initHistory();
});

function initHistory() {
    document.getElementById('clearHistoryBtn').addEventListener('click', () => {
        sessionHistory = [];
        renderHistory();
    });
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('historyList');

    if (sessionHistory.length === 0) {
        list.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 2rem 0;">No history yet</div>';
        return;
    }

    list.innerHTML = sessionHistory.map((item, index) => `
        <div class="history-item" onclick="loadHistoryItem(${index})">
            <img src="${item.thumbnail}" class="history-thumbnail" alt="Thumb">
            <div class="history-info">
                <div class="history-label">${item.prediction.class_name}</div>
                <div class="history-meta">${new Date(item.timestamp).toLocaleTimeString()} • ${item.method}</div>
            </div>
            <div style="font-weight: bold; color: var(--success-color); font-size: 0.8rem;">
                ${(item.prediction.probability * 100).toFixed(0)}%
            </div>
        </div>
    `).join('');
}

function loadHistoryItem(index) {
    const item = sessionHistory[index];
    if (!item) return;

    // Restore image preview first (crucial for displayResults which uses src)
    previewImage.src = item.thumbnail;
    imagePreview.classList.remove('hidden');
    analysisControls.classList.remove('hidden');

    // Restore results
    displayResults(item.fullPrediction, item.explanations);

    // Scroll to top of results or preview
    imagePreview.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Update progress bar
 */
function updateProgress(percent, message) {
    const validPercent = Math.min(Math.max(0, percent), 100);
    const progressDiv = document.getElementById('analysisProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressPercent = document.getElementById('progressPercent');

    progressDiv.classList.remove('hidden');
    requestAnimationFrame(() => {
        progressBar.style.width = `${validPercent}%`;
        if (message) progressText.textContent = message;
        progressPercent.textContent = `${Math.round(validPercent)}%`;
    });
}

/**
 * Initialize tab switching
 */
function initTabs() {
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            // Update button states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update content visibility
            tabContents.forEach(content => {
                if (content.id === `${targetTab}-tab`) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });
}

/**
 * Initialize gallery with dynamic content
 */
function initGallery() {
    examplesGallery.innerHTML = EXAMPLES.map(example => `
        <div class="gallery-item" onclick="loadExample('${example.id}')">
            <div class="gallery-image-container">
                <img src="${API_BASE_URL}/assets/precomputed/${example.id}/original.jpg" 
                     alt="${example.name}" loading="lazy"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dywudzMub3JnLzIwMDAvc3ZnIiB2aWV3Qm94PSIwIDAgMjQgMjUiPjxwYXRoIGZpbGw9IiNjY2MiIGQ9M00zIDNoMTh2MThIM1oiLz48L3N2Zz4='">
                <div class="gallery-overlay"></div>
            </div>
            <div class="gallery-content">
                <div class="gallery-label">${example.name}</div>
                <div class="gallery-meta">
                    <span class="gallery-badge">Precomputed</span>
                </div>
            </div>
        </div>
    `).join('');

    backToGallery.addEventListener('click', () => {
        exampleDetail.classList.add('hidden');
        examplesGallery.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

/**
 * Load and display a precomputed example
 */
async function loadExample(exampleId) {
    // Show detail view
    examplesGallery.classList.add('hidden');
    exampleDetail.classList.remove('hidden');
    exampleDetail.scrollIntoView({ behavior: 'smooth' });

    const example = EXAMPLES.find(e => e.id === exampleId);
    if (!example) return;

    // Set Header Info
    document.getElementById('exampleTitle').textContent = example.name;
    document.getElementById('exampleSubtitle').textContent = `Example analysis of ${example.name.toLowerCase()}`;
    document.getElementById('exampleConfidence').textContent = "Loading...";

    // Fetch metadata
    try {
        const response = await fetch(`${API_BASE_URL}/assets/precomputed/${exampleId}/metadata.json`);
        const metadata = await response.json();

        const topPred = metadata.top_prediction;
        document.getElementById('exampleConfidence').textContent =
            `${topPred.quantity || (topPred.probability * 100).toFixed(1)}% Confidence`;

    } catch (e) {
        console.error("Could not load metadata", e);
        document.getElementById('exampleConfidence').textContent = "Analysis Ready";
    }

    // Populate Visualizations
    const basePath = `${API_BASE_URL}/assets/precomputed/${exampleId}`;

    const vizItems = [
        { key: 'original', img: 'original.jpg' },
        { key: 'gradcam', img: 'gradcam_vis.jpg' },
        { key: 'occlusion', img: 'occlusion_vis.jpg' },
        { key: 'guidedbackprop', img: 'guided_backprop.jpg' },
        { key: 'guided_gradcam', img: 'guided_gradcam.jpg' }
    ];

    const container = document.getElementById('exampleResults');
    container.innerHTML = vizItems.map(item => {
        const info = EXPLANATIONS[item.key] || { name: item.key, icon: '', desc: '' };
        return `
        <div class="viz-card">
            <div class="viz-content-side">
                <div class="viz-header">
                    <div class="viz-title">${info.name}</div>
                </div>
                <div class="viz-description">
                    ${info.desc}
                </div>
            </div>
            <div class="viz-image-container">
                <img src="${basePath}/${item.img}" alt="${info.name}" loading="lazy">
            </div>
        </div>
        `;
    }).join('');
}

/**
 * Initialize file upload functionality
 */
function initUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
}

/**
 * Handle uploaded file
 */
function handleFile(file) {
    currentFile = file;

    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.classList.remove('hidden');
        analysisControls.classList.remove('hidden');
        interactiveResults.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

/**
 * Initialize analyze functionality
 */
function initAnalyze() {
    analyzeBtn.addEventListener('click', analyzeImage);
}

/**
 * Analyze uploaded image
 */
async function analyzeImage() {
    if (!currentFile) {
        alert('Please upload an image first');
        return;
    }

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    updateProgress(0, 'Initializing...');

    try {
        // Step 1: Upload & Prediction
        // Simulate upload progress
        updateProgress(10, 'Uploading image...');
        await new Promise(r => setTimeout(r, 500));

        updateProgress(30, 'Running prediction model...');
        const prediction = await getPrediction(currentFile);

        // Step 2: Explanations
        updateProgress(50, 'Generating explanations...');
        const method = document.getElementById('methodSelect').value;
        let explanations = {};

        if (method === 'all') {
            // Get all methods concurrently
            // We can break this down to update progress for each
            const tasks = [
                { key: 'gradcam', p: 60, name: 'Grad-CAM' },
                { key: 'guidedbackprop', p: 70, name: 'Guided Backprop' },
                { key: 'guided_gradcam', p: 80, name: 'Guided Grad-CAM' },
                { key: 'occlusion', p: 95, name: 'Occlusion (this takes longest)' }
            ];

            // Start all promises but update progress as they finish?
            // Sequential is better for progress bar visualization

            // Grad-CAM
            updateProgress(60, 'Generating Grad-CAM...');
            explanations.gradcam = await getExplanation('gradcam', currentFile);

            // Guided Backprop
            updateProgress(70, 'Generating Guided Backprop...');
            explanations.guidedbackprop = await getExplanation('guidedbackprop', currentFile);

            // Guided Grad-CAM
            updateProgress(80, 'Generating Guided Grad-CAM...');
            explanations.guided_gradcam = await getExplanation('guided_gradcam', currentFile);

            // Occlusion (Parallelizes internally in backend if possible, but request is one block)
            updateProgress(85, 'Running Occlusion Sensitivity...');
            explanations.occlusion = await getExplanation('occlusion', currentFile);

        } else {
            updateProgress(70, `Running ${method}...`);
            explanations[method] = await getExplanation(method, currentFile);
        }

        updateProgress(100, 'Complete!');

        // Display results
        displayResults(prediction, explanations);

        // Save to History
        sessionHistory.unshift({
            timestamp: Date.now(),
            thumbnail: previewImage.src,
            prediction: prediction.top_prediction,
            fullPrediction: prediction,
            explanations: explanations,
            method: method === 'all' ? 'All Methods' : EXPLANATIONS[method].name
        });
        renderHistory();

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed. Please make sure the backend is running on http://localhost:8000');
        updateProgress(0, 'Failed');
    } finally {
        // Reset button
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '🔍 Analyze Image';
        // Hide progress after a delay
        setTimeout(() => {
            // document.getElementById('analysisProgress').classList.add('hidden');
        }, 3000);
    }
}

/**
 * Get prediction from API
 */
async function getPrediction(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', '5');

    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

/**
 * Get explanation from API
 */
async function getExplanation(method, file) {
    const formData = new FormData();
    formData.append('file', file);

    if (method === 'occlusion') {
        const windowSize = document.getElementById('windowSize').value;
        formData.append('window_size', windowSize);
        formData.append('stride', Math.floor(windowSize / 2));
    }

    const response = await fetch(`${API_BASE_URL}/explain/${method}`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

/**
 * Display analysis results
 */
function displayResults(prediction, explanations) {
    // Show results section
    interactiveResults.classList.remove('hidden');

    // Update prediction
    const topPred = prediction.top_prediction;
    document.getElementById('predictionLabel').textContent = topPred.class_name;
    document.getElementById('predictionConfidence').textContent =
        `Confidence: ${(topPred.probability * 100).toFixed(2)}%`;

    // Update top predictions
    const topPredictionsHTML = prediction.top_k_predictions.map(pred => `
        <div style="margin-bottom: 1rem;">
            <div class="prediction-item">
                <span>${pred.class_name}</span>
                <span>${(pred.probability * 100).toFixed(2)}%</span>
            </div>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: ${pred.probability * 100}%"></div>
            </div>
        </div>
    `).join('');
    document.getElementById('topPredictions').innerHTML = topPredictionsHTML;

    // Use the same horizontal card style

    // Add original image first
    let html = `
    <div class="viz-card">
        <div class="viz-content-side">
            <div class="viz-header">
                <div class="viz-icon">🖼️</div>
                <div class="viz-title">Original Image</div>
            </div>
            <div class="viz-description">
                ${EXPLANATIONS.original.desc}
            </div>
        </div>
        <div class="viz-image-container">
            <img src="${previewImage.src}" alt="Original">
        </div>
    </div>
    `;

    // Add explanations
    html += Object.entries(explanations).map(([method, data]) => {
        // Map method keys (gradcam -> gradcam, occlusion -> occlusion, guidedbackprop -> guidedbackprop, guided_gradcam -> guided_gradcam)
        // Ensure keys match EXPLANATIONS keys
        let key = method;
        const info = EXPLANATIONS[key] || { name: method, icon: '❓', desc: '' };
        const mainImage = data.visualization; // Base64

        return `
        <div class="viz-card">
            <div class="viz-content-side">
                <div class="viz-header">
                    <div class="viz-icon">${info.icon}</div>
                    <div class="viz-title">${info.name}</div>
                </div>
                <div class="viz-description">
                    ${info.desc}
                </div>
            </div>
            <div class="viz-image-container">
                <img src="${mainImage}" alt="${info.name}">
            </div>
        </div>
        `;
    }).join('');

    document.getElementById('explanationsGrid').innerHTML = html;

    // Scroll to results
    interactiveResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        getPrediction,
        getExplanation,
        loadExample
    };
}
