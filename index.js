const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors()); // Enable CORS for all routes

app.post('/analyze', (req, res) => {
    const { sentence } = req.body;
    axios.post('http://127.0.0.1:8080/predict', { sentence })
        .then((response) => {
            const modifiedData = response.data.result;
            res.json({ modifiedData });
        })
        .catch((error) => {
            console.error('Error:', error);
            res.status(500).json({ error: 'Something went wrong' });
        });
});

app.listen(3005, () => {
    console.log('Server is listening on port 3005');
});
