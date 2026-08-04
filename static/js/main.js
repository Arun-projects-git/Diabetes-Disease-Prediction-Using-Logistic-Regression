document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // Tab Switching Logic
    // -------------------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'analytics-tab') {
                fetchModelMetrics();
            }
        });
    });

    // -------------------------------------------------------------
    // Predictor Form Submission
    // -------------------------------------------------------------
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const recommendationsList = document.getElementById('recommendations-list');
    const riskBadge = document.getElementById('risk-badge');
    const riskBadgeIcon = document.getElementById('risk-badge-icon');
    const riskBadgeText = document.getElementById('risk-badge-text');
    const probabilityValue = document.getElementById('probability-value');
    const circularProgress = document.querySelector('.circular-progress');
    const imputationBanner = document.getElementById('imputation-banner');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        // Gather form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Send empty strings as is, so server can impute if they are blank
            if (value.trim() === '') {
                data[key] = '';
            } else {
                data[key] = parseFloat(value);
            }
        });

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.status === 'success') {
                displayPrediction(result);
            } else {
                alert(`Error: ${result.message}`);
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            alert('An error occurred while running the prediction. Please make sure the Flask backend is running and the model has been trained.');
        } finally {
            // Reset loading state
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    function displayPrediction(data) {
        // Activate result card, hiding the placeholder
        resultContainer.classList.remove('empty');
        resultContainer.classList.add('active');

        // Set risk badge
        riskBadge.className = 'risk-badge'; // reset
        if (data.prediction === 1) {
            riskBadge.classList.add('high');
            riskBadgeText.textContent = 'High Risk';
            riskBadgeIcon.className = 'fa-solid fa-triangle-exclamation';
        } else {
            riskBadge.classList.add('low');
            riskBadgeText.textContent = 'Low Risk';
            riskBadgeIcon.className = 'fa-solid fa-circle-check';
        }

        // Animate circular progress wheel
        const targetPercent = Math.round(data.probability * 100);
        animateProgress(targetPercent, data.prediction);

        // Show/hide imputation warning banner
        const imputedKeys = Object.keys(data.imputed_values);
        if (imputedKeys.length > 0) {
            imputationBanner.style.display = 'flex';
            imputationBanner.querySelector('span').textContent = `Note: Zero or empty inputs for [${imputedKeys.join(', ')}] were auto-filled with median training values: ${JSON.stringify(data.imputed_values)}`;
        } else {
            imputationBanner.style.display = 'none';
        }

        // Populate clinical suggestions
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });

        // Scroll to results on mobile devices
        if (window.innerWidth <= 900) {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function animateProgress(targetValue, predictionClass) {
        let startValue = 0;
        const duration = 800; // ms
        const stepTime = Math.abs(Math.floor(duration / targetValue || 1));
        
        // Pick progress ring color based on risk class
        const progressColor = predictionClass === 1 ? '#ef4444' : '#10b981';

        const timer = setInterval(() => {
            if (startValue >= targetValue) {
                clearInterval(timer);
            }
            probabilityValue.textContent = `${startValue}%`;
            circularProgress.style.background = `conic-gradient(${progressColor} ${startValue * 3.6}deg, rgba(255,255,255,0.05) 0deg)`;
            
            if (startValue < targetValue) {
                startValue++;
            }
        }, stepTime);
    }

    // -------------------------------------------------------------
    // Analytics Metrics Fetching
    // -------------------------------------------------------------
    async function fetchModelMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const result = await response.json();

            if (result.status === 'success') {
                const metrics = result.data;
                document.getElementById('metric-accuracy').textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
                document.getElementById('metric-precision').textContent = `${(metrics.precision * 100).toFixed(1)}%`;
                document.getElementById('metric-recall').textContent = `${(metrics.recall * 100).toFixed(1)}%`;
                document.getElementById('metric-f1').textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
                document.getElementById('metric-auc').textContent = metrics.roc_auc.toFixed(3);
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
            // Metrics might not be available yet if pipeline hasn't been run
        }
    }
});
