async function analyzeSpam() {
    const text = document.getElementById('emailInput').value;
    const resultCard = document.getElementById('resultCard');
    const resultTitle = document.getElementById('resultTitle');
    const resultDesc = document.getElementById('resultDesc');
    const statusIcon = document.getElementById('statusIcon');
    const confidenceScore = document.getElementById('confidenceScore');

    if (!text.trim()) {
        alert("Please enter some text to analyze.");
        return;
    }

    // Show loading state could be here
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ content: text }),
        });

        const data = await response.json();

        resultCard.classList.remove('hidden');
        resultCard.classList.remove('is-spam', 'is-safe');

        if (data.is_spam) {
            resultCard.classList.add('is-spam');
            resultTitle.textContent = "SPAM DETECTED";
            resultDesc.textContent = "This email has been flagged as potential spam.";
            statusIcon.className = "fas fa-exclamation-circle";
        } else {
            resultCard.classList.add('is-safe');
            resultTitle.textContent = "SAFE MESSAGE";
            resultDesc.textContent = "This email appears to be legitimate (Ham).";
            statusIcon.className = "fas fa-check-circle";
        }

        confidenceScore.textContent = data.confidence;

    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred while analyzing the text.");
    }
}

function clearText() {
    document.getElementById('emailInput').value = '';
    document.getElementById('resultCard').classList.add('hidden');
}
