var express = require('express'); // Express web server framework
var app = express();
app.use(express.static(__dirname));
console.log('Listening on 8887');
app.listen(8887);
