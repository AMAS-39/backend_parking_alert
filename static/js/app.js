document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('uploadForm');
    const resultMessage = document.getElementById('resultMessage');
    const logList = document.getElementById('logList');

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/process_image', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                resultMessage.textContent = `Plate Number Detected: ${data.plate_number}`;
                resultMessage.style.display = 'block';
                addLog(data.plate_number);
            } else {
                resultMessage.textContent = `Error: ${data.message}`;
                resultMessage.style.display = 'block';
            }
        } catch (error) {
            resultMessage.textContent = 'An error occurred while processing the image.';
            resultMessage.style.display = 'block';
        }
    });

    function addLog(plateNumber) {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        listItem.textContent = `Detected Plate: ${plateNumber}`;
        logList.prepend(listItem);
    }
});
