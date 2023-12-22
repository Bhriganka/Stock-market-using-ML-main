const express = require('express');
const app = express();
const cors = require('cors');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => res.send('Hello World!'));

app.get('/predict', (req, res) => {
  const options = {
    // mode: 'text',
    // pythonOptions: ['-u'], // unbuffered output,
    args:[req.query.stock]
    // args:["MSFT"]
  };

  PythonShell.run('index2.py', options).then(result => {
    console.log(result)
    res.send(result)

  }).catch(err => {
    res.send(err);
  })
});
app.get('/predict2', (req, res) => {
  const options = {
    // mode: 'text',
    // pythonOptions: ['-u'], // unbuffered output,
    // args:[req.query.stock]
    args:["MSFT"]
  };

  PythonShell.run('index3.py', null).then(result => {
    console.log(result)
    res.send(result)

  }).catch(err => {
    res.send(err);
  })
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`));



 // const tableStartIndex = result.indexOf('Open         High          Low        Close   ');
    // const tableEndIndex = result.indexOf('[7425 rows x 17 columns]');

    // const tableData = result.slice(tableStartIndex + 2, tableEndIndex);
    // const formattedData = tableData.map(row => row.trim());
    // console.log(formattedData);
    // res.send(formattedData)
 // const tableStartIndex = result.indexOf('Open         High          Low        Close   ');
    // const tableEndIndex = result.indexOf('[7425 rows x 17 columns]');

    // const tableData = result.slice(tableStartIndex + 2, tableEndIndex);
    // const table = tableData.splice(1)

    // console.log(table)
    // res.send(table);