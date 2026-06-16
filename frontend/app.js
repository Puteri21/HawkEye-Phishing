// =============================================================
// FILE: frontend/app.js
// PURPOSE: This is the WEBSITE side of the connection.
//          It sends user input to the FastAPI and receives
//          the ML model prediction result to display on screen.
//
// CONNECTION FLOW:
//   User types text --> app.js --> FastAPI (api/main.py) --> ML Model
//   Website shows   <-- app.js <-- FastAPI sends JSON    <-- Result
// =============================================================

// This is the ADDRESS of the FastAPI server (the bridge/middleman)
// FastAPI must be running (uvicorn) for this to work
const API_BASE_URL = 'http://127.0.0.1:8000';

// Tab Switching Logic
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Hide result box when switching tabs
        document.getElementById('result-container').classList.add('hidden');

        // Add active class to clicked button and target content
        button.classList.add('active');
        const targetId = button.getAttribute('data-target');
        document.getElementById(targetId).classList.add('active');
    });
});

// Clear Input Logic
function clearInput(type) {
    let inputElementId = '';
    if (type === 'email') inputElementId = 'email-input';
    else if (type === 'sms') inputElementId = 'sms-input';
    else if (type === 'url') inputElementId = 'url-input';

    // Clear the textarea/input
    document.getElementById(inputElementId).value = '';
    
    // Hide the result box
    document.getElementById('result-container').classList.add('hidden');
}

async function analyze(type) {
    let inputElementId = '';
    let endpoint = '';

    // Set the correct API endpoint based on which tab user is on
    // These must match the endpoints defined in api/main.py
    if (type === 'email') {
        inputElementId = 'email-input';
        endpoint = '/predict/email';  // --> calls api/main.py @app.post("/predict/email")
    } else if (type === 'sms') {
        inputElementId = 'sms-input';
        endpoint = '/predict/sms';    // --> calls api/main.py @app.post("/predict/sms")
    } else if (type === 'url') {
        inputElementId = 'url-input';
        endpoint = '/predict/url';    // --> calls api/main.py @app.post("/predict/url")
    }

    const inputElement = document.getElementById(inputElementId);
    const text = inputElement.value.trim();

    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }

    // UI Updates - Show Loading
    const resultBox = document.getElementById('result-container');
    const loaderContainer = document.getElementById('loader-container');
    const resultContent = document.getElementById('result-content');
    const confidenceFill = document.getElementById('confidence-fill');
    const keywordsContainer = document.getElementById('keywords-container');
    const keywordsList = document.getElementById('keywords-list');

    // Reset classes
    resultBox.className = 'result-box';
    confidenceFill.style.width = '0%';
    keywordsContainer.classList.add('hidden');
    keywordsList.innerHTML = '';

    resultBox.classList.remove('hidden');
    loaderContainer.classList.remove('hidden');

    // Change loader text for effect
    const scanStages = ['Extracting Features...', 'Running BHHO Analysis...', 'Calculating Confidence...'];
    let stage = 0;
    const scanInterval = setInterval(() => {
        if (stage < scanStages.length) {
            loaderContainer.setAttribute('data-text', scanStages[stage]);
            stage++;
        }
    }, 500);

    resultContent.classList.add('hidden');

    try {
        // -----------------------------------------------------------
        // THIS IS WHERE WEBSITE CONNECTS TO THE ML MODEL via FastAPI
        // fetch() sends an HTTP POST request to the FastAPI server.
        // It sends the user's text as JSON to the API address.
        //
        // Example request sent:
        //   POST http://127.0.0.1:8000/predict/email
        //   Body: { "text": "Click here to verify your account" }
        // -----------------------------------------------------------
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',                          // type of request
            headers: {
                'Content-Type': 'application/json',  // tell API we are sending JSON
            },
            body: JSON.stringify({ text: text })      // convert text to JSON format
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze.');
        }

        // -----------------------------------------------------------
        // RECEIVE THE RESULT FROM FASTAPI (ML Model output)
        // data is the JSON sent back from api/main.py
        // It contains: prediction, confidence, is_phishing, flagged_keywords
        //
        // Example result received:
        //   { "prediction": "Phishing",
        //     "confidence": 91.5,
        //     "is_phishing": true,
        //     "flagged_keywords": ["verify", "click", "account"] }
        // -----------------------------------------------------------
        const data = await response.json();

        // Simulate a slight delay for deep scan animation effect
        setTimeout(() => {
            clearInterval(scanInterval);
            loaderContainer.classList.add('hidden');
            resultContent.classList.remove('hidden');

            // Display prediction text ("Phishing" or "Legitimate") from ML model
            document.getElementById('prediction-text').innerText = data.prediction;
            // Display confidence score (%) from ML model
           document.getElementById('confidence-value').innerText = data.confidence.toFixed(2);

            // Set style based on result
            if (data.is_phishing) {
                resultBox.classList.add('phishing');

                // Show keywords if they exist
                if (data.flagged_keywords && data.flagged_keywords.length > 0) {
                    keywordsContainer.classList.remove('hidden');
                    data.flagged_keywords.forEach(kw => {
                        const span = document.createElement('span');
                        span.className = 'keyword-tag';
                        span.innerText = kw;
                        keywordsList.appendChild(span);
                    });
                }
            } else {
                resultBox.classList.add('safe');
            }

            // Animate confidence bar
            setTimeout(() => {
                confidenceFill.style.width = `${data.confidence}%`;
            }, 50);

        }, 1500); // 1.5 seconds scanning simulation

    } catch (error) {
        clearInterval(scanInterval);
        loaderContainer.classList.add('hidden');
        alert(`Error: ${error.message}`);
        resultBox.classList.add('hidden');
    }
}

// Modal logic removed
