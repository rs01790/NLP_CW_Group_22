const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { analyzeEmotion } = require('./analyzer');

const app = express();

app.use(cors());
app.use(bodyParser.json());

app.post('/analyze', (req, res) => {
    const { sentence } = req.body;
    const result = analyzeEmotion(sentence);
    res.json({ result });
});

app.listen(3005, () => {
    console.log('Server is listening on port 3005');
});