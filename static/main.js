document.getElementById('searchForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const form = document.getElementById('searchForm');
    const formData = new FormData(form);

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    try {
        const response = await fetch('/search', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log(data);

        if (data.error) {
            resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            displayResults(data);
        }
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = `<p>Something went wrong. Please try again later.</p>`;
    }
});


function displayResults(data) {
    const resultsDiv = document.getElementById('results');

    // Clear any previous results
    resultsDiv.innerHTML = '';

    // Check if the data contains image paths and similarities
    if (!data.image_paths || !data.similarities || data.image_paths.length === 0) {
        resultsDiv.innerHTML = '<p>No results found.</p>';
        return;
    }

    // Loop through the image paths and similarities
    data.image_paths.forEach((imagePath, index) => {
        // Create a container for each result
        const resultContainer = document.createElement('div');
        resultContainer.className = 'result-container';

        // Create an img element for the image
        const img = document.createElement('img');
        img.src = imagePath; // Set the image source
        img.alt = `Result ${index + 1}`; // Set alt text for accessibility
        img.className = 'result-image';

        // Create a paragraph element for the similarity
        const similarityText = document.createElement('p');
        similarityText.textContent = `Similarity: ${data.similarities[index].toFixed(2)}`;

        // Append the image and similarity text to the result container
        resultContainer.appendChild(img);
        resultContainer.appendChild(similarityText);

        // Append the result container to the results div
        resultsDiv.appendChild(resultContainer);
    });
}

