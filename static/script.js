document.addEventListener('DOMContentLoaded', () => {
    const tweetInput = document.getElementById('tweetInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const charCount = document.querySelector('.char-count');
    const sentimentBox = document.getElementById('sentimentBox');
    const sentimentText = document.getElementById('sentiment');
    const confidenceText = document.getElementById('confidence');

    // Character counter
    tweetInput.addEventListener('input', () => {
        const count = tweetInput.value.length;
        charCount.textContent = `${count}/280`;
    });

    // Analyze button click
    analyzeBtn.addEventListener('click', async () => {
        const tweet = tweetInput.value.trim();
        
        if (!tweet) {
            alert('Please enter a tweet!');
            return;
        }

        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;

        try {
            console.log('Sending request with tweet:', tweet);
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tweet }),
            });

            const data = await response.json();
            console.log('Response data:', data);

            if (data.success) {
                sentimentText.textContent = `Sentiment: ${data.sentiment}`;
                confidenceText.textContent = `Confidence: ${data.confidence}%`;
                
                sentimentBox.className = 'sentiment-box';
                sentimentBox.classList.add(data.sentiment.toLowerCase());
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            sentimentText.textContent = 'Error';
            confidenceText.textContent = error.message;
            sentimentBox.className = 'sentiment-box';
        }

        analyzeBtn.textContent = 'Analyze Sentiment';
        analyzeBtn.disabled = false;
    });
});